# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/11_train.ipynb (unless otherwise specified).

__all__ = ['DataLoop', 'BeforeForward', 'Forward', 'BeforeLoss', 'BeforeLoss']

# Cell
from .loop import Loop,Event,method4all,StorageCore

class DataLoop(Loop):
    def __init__(self,dataloader):
        super().__init__(iterable=dataloader)

class BeforeForward(Event):
    def __init__(self,):
        super().__init__(self,event_name = "before_forward")

class Forward(Event):
    def __init__(self,):
        super().__init__(self,event_name = "forward")

class BeforeLoss(Event):
    def __init__(self,):
        super().__init__(self,event_name = "before_loss")

class BeforeLoss(Event):
    def __init__(self,):
        super().__init__(self,event_name = "before_loss")