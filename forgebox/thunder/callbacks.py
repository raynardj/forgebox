import pandas as pd
from ipywidgets import Output
from typing import List
import copy
try:
    import pytorch_lightning as pl
except ImportError as e:
    raise ImportError("Please install pytorch-lightning first")
from forgebox.html import display
from torch import nn


def unfreeze(self):
    """unfreeze this module, and its sub modules"""
    for p in self.parameters():
        p.requires_grad = True


def freeze(self):
    """freeze this module, and its sub modules"""
    for p in self.parameters():
        p.requires_grad = False


nn.Module.unfreeze = unfreeze
nn.Module.freeze = freeze


class DataFrameMetricsCallback(pl.Callback):
    """
    A metrics callback keep showing pandas dataframe
    """

    def __init__(self) -> None:
        """
        In Trainer kwargs, passing this arguements along with other callbacks
        callbacks = [DataFrameMetricsCallback(),]
        """
        self.metrics: List = []

    def on_fit_start(
        self, trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        pl_module.output = Output()
        display(pl_module.output)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        self.metrics.append(dict((k, v.item())
                                 for k, v in metrics_dict.items()))
        pl_module.output.clear_output()
        with pl_module.output:
            display(pd.DataFrame(self.metrics).tail(10))
