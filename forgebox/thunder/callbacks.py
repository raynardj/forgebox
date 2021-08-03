# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/61_thunder_callbacks.ipynb (unless otherwise specified).

__all__ = ['unfreeze', 'freeze', 'DataFrameMetricsCallback', 'UnfreezeScheduler']

# Cell
import pandas as pd
from ipywidgets import Output
from typing import List, Dict
import copy
import pytorch_lightning as pl
import torch
from torch import nn

# Cell
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


def UnfreezeScheduler(frozen_epochs: int = 2):
    assert hasattr(pl_module, "top_layers"), "Please define 'top_layers' attributes"+\
    " for pl_module, which will return a list of nn.Module object(s)"
    class UnfreezeSchedulerCallback(pl.callbacks.Callback):
        """
        Train the top layer for [frozen_epochs] epochs
        then un freeze all
        """

        def on_epoch_start(self, trainer, pl_module):
            epoch = trainer.current_epoch

            if epoch == 0:
                pl_module.freeze()
                for tl in pl_module.top_layers:
                    tl.unfreeze()
            if epoch == frozen_epochs:
                pl_module.unfreeze()
                pl_module.base.embeddings.freeze()