import os
from typing import Iterable

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        for arg in ["num_labels", "task_name"]:
            parser.link_arguments(
                f"data.init_args.{arg}",
                f"model.init_args.{arg}",
                apply_on="instantiate",
            )

    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand]

        logger = config.trainer.logger
        if logger and logger is not True:
            loggers = logger if isinstance(logger, Iterable) else [logger]
            for logger in loggers:
                logger.init_args.save_dir = os.path.join(
                    logger.init_args.get("save_dir", "results"), self.subcommand
                )


def lit_cli():
    LitCLI(
        parser_kwargs={
            cmd: {
                "default_config_files": ["configs/presets/default.yaml"],
            }
            for cmd in ["fit", "validate", "test"]
        },
        save_config_kwargs={"overwrite": True},
    )


def get_cli_parser():
    # provide cli.parser for shtab.
    #
    # install shtab in the same env and run
    # shtab --shell {bash,zsh,tcsh} src.utils.lit_cli.get_cli_parser
    # for more details see https://docs.iterative.ai/shtab/use/#cli-usage
    from jsonargparse import capture_parser

    from . import tweak_shtab  # noqa

    parser = capture_parser(lit_cli)
    return parser


if __name__ == "__main__":
    lit_cli()
