import os

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


# TODO:
# https://github.com/Lightning-AI/lightning/issues/14188
# https://github.com/Lightning-AI/lightning/pull/14640
@property
def log_dir(self) -> str:
    if self.loggers and self.loggers[0].log_dir is not None:
        dirpath = self.loggers[0].log_dir
    else:
        dirpath = self.default_root_dir

    dirpath = self.strategy.broadcast(dirpath)
    return dirpath


pl.Trainer.log_dir = log_dir


def __resolve_ckpt_dir(self, trainer: pl.Trainer) -> _PATH:
    """Determines model checkpoint save directory at runtime. References attributes from the trainer's logger
    to determine where to save checkpoints. The base path for saving weights is set in this priority:
    1.  Checkpoint callback's path (if passed in)
    2.  The default_root_dir from trainer if trainer has no logger
    3.  The log_dir from trainer, if trainer has logger
    """
    if self.dirpath is not None:
        # short circuit if dirpath was passed to ModelCheckpoint
        return self.dirpath
    if trainer.loggers:
        ckpt_path = os.path.join(trainer.log_dir, "checkpoints")
    else:
        ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")
    return ckpt_path


ModelCheckpoint._ModelCheckpoint__resolve_ckpt_dir = __resolve_ckpt_dir


@property
def WandbLogger_log_dir(self) -> str:
    if hasattr(self, "_log_dir"):
        return self._log_dir

    save_dir = self._save_dir
    name = self.experiment.name
    version = self.experiment.id
    try:
        self._log_dir = os.path.join(save_dir, name, version)
    except TypeError:
        return None

    return self._log_dir


WandbLogger.log_dir = WandbLogger_log_dir
