import itertools
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import ray
from jsonargparse import CLI
from ray import train, tune

from src.utils.lit_cli import lit_cli

ray.init(
    _temp_dir=str(Path.home() / ".cache" / "ray"), num_cpus=min(os.cpu_count(), 32)
)


def run_cli(config, debug: bool = True, command: str = "fit", devices: int = 1):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    argv = ["./run", command]
    ckpt_path = config.pop("ckpt_path", None)
    if ckpt_path is not None:
        config_path = Path(ckpt_path).parents[1] / "config.yaml"
        argv.extend(["--config", str(config_path)])
        argv.extend(["--ckpt_path", ckpt_path])
        config.pop("config", None)
        config.pop("data_config", None)
    else:
        for cfg in ["config", "data_config"]:
            if cfg in config:
                argv.extend(["--config", config.pop(cfg)])

    argv.extend(
        itertools.chain(
            *[
                [f"--{k}", v if isinstance(v, str) else json.dumps(v)]
                for k, v in config.items()
            ]
        )
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
    gpus_per_trial: int | float = 1,
    *,
    ckpt_paths: list[str | None] | None = None,
    configs: list[str] | None = None,
    data_configs: list[str | None] | None = None,
    override_kwargs: dict[str, Any] | None = None,
):
    param_space = {
        **({"ckpt_path": tune.grid_search(ckpt_paths)} if ckpt_paths else {}),
        **({"config": tune.grid_search(configs)} if configs else {}),
        **({"data_config": tune.grid_search(data_configs)} if data_configs else {}),
        **(
            {
                k: tune.grid_search(v) if isinstance(v, list) else tune.grid_search([v])
                for k, v in override_kwargs.items()
            }
            if override_kwargs
            else {}
        ),
    }

    tune_config = tune.TuneConfig()
    run_config = train.RunConfig(
        log_to_file=True,
        storage_path=Path("./results/ray").resolve(),
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
    sweep("fit", *args, **kwargs)


def validate(*args, **kwargs):
    sweep("validate", *args, **kwargs)


def test(*args, **kwargs):
    sweep("test", *args, **kwargs)


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
