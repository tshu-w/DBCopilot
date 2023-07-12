import itertools
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any, Literal, Optional, Union
from unittest.mock import patch

import ray
from jsonargparse import CLI
from ray import air, tune

from src.utils.lit_cli import lit_cli

os.environ["PL_DISABLE_FORK"] = "1"
ray.init(_temp_dir=str(Path.home() / ".cache" / "ray"))


def run_cli(config, debug: bool = True, command: str = "fit", devices: int = 1):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    argv = ["./run", command]
    argv.extend(["--config", config.pop("config")])

    data_config = config.pop("data_config", None)
    if data_config:
        argv.extend(["--config", data_config])

    argv.extend(
        itertools.chain(*[[f"--{k}", json.dumps(v)] for k, v in config.items()])
    )

    argv.extend(["--trainer.devices", str(devices)])
    if debug:
        argv.extend(["--config", "configs/presets/tester.yaml"])

    print(shlex.join(argv))
    with patch("sys.argv", argv):
        lit_cli()

    try:
        import wandb

        wandb.finish()
    except ImportError:
        pass


def sweep(
    command: Literal["fit", "validate", "test"],
    debug: bool = False,
    gpus_per_trial: Union[int, float] = 1,
    *,
    configs: list[str] = ["configs/mnist.yaml"],
    data_configs: list[Optional[str]] = [None],
    override_kwargs: dict[str, Any] = {},
):
    param_space = {
        "config": tune.grid_search(configs),
        "data_config": tune.grid_search(data_configs),
        **{
            k: tune.grid_search(v) if isinstance(v, list) else tune.grid_search([v])
            for k, v in override_kwargs.items()
        },
    }

    tune_config = tune.TuneConfig()
    run_config = air.RunConfig(
        log_to_file=True,
        storage_path="results/ray",
        verbose=1,
    )
    trainable = tune.with_parameters(
        run_cli,
        debug=debug,
        command=command,
        devices=math.ceil(gpus_per_trial),
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()


def fit(*args, **kwargs):
    sweep(command="fit", *args, **kwargs)


def validate(*args, **kwargs):
    sweep(command="validate", *args, **kwargs)


def test(*args, **kwargs):
    sweep(command="test", *args, **kwargs)


def sweep_cli():
    CLI([fit, validate, test])


def get_cli_parser():
    # provide cli.parser for shtab.
    #
    # install shtab in the same env and run
    # shtab --shell {bash,zsh,tcsh} src.utils.lit_cli.get_cli_parser
    # for more details see https://docs.iterative.ai/shtab/use/#cli-usage
    from jsonargparse import capture_parser

    from . import tweak_shtab  # noqa

    parser = capture_parser(sweep_cli)
    return parser


if __name__ == "__main__":
    sweep_cli()
