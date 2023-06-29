import json
import logging
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning.pytorch.trainer.states import TrainerFn


class Metric(pl.Callback):
    r"""
    Save logged metrics to ``Trainer.log_dir``.
    """

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        metrics = {}
        if stage == TrainerFn.FITTING:
            if (
                trainer.checkpoint_callback
                and trainer.checkpoint_callback.best_model_path
            ):
                ckpt_path = trainer.checkpoint_callback.best_model_path
                # inhibit disturbing logging
                logging.getLogger("lightning.pytorch.utilities.distributed").setLevel(
                    logging.WARNING
                )
                logging.getLogger("lightning.pytorch.accelerators.gpu").setLevel(
                    logging.WARNING
                )

                fn_kwargs = {
                    "model": pl_module,
                    "datamodule": trainer.datamodule,
                    "ckpt_path": ckpt_path,
                }

                val_metrics = {}
                if trainer.validate_loop._data_source.is_defined():
                    trainer.callbacks = []
                    trainer.validate(**fn_kwargs)
                    val_metrics = convert_tensors_to_scalars(trainer.logged_metrics)

                test_metrics = {}
                if trainer.test_loop._data_source.is_defined():
                    trainer.callbacks = []
                    trainer.test(**fn_kwargs)
                    test_metrics = convert_tensors_to_scalars(trainer.logged_metrics)

                metrics = {**val_metrics, **test_metrics}
        else:
            metrics = convert_tensors_to_scalars(trainer.logged_metrics)

        if metrics:
            metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2)

            metrics_file = Path(trainer.log_dir) / "metrics.json"
            with metrics_file.open("w") as f:
                f.write(metrics_str)
