# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_loopstack.ipynb (unless otherwise specified).

__all__ = ['create_event', 'events', 'LoopStack', 'train_callbacks', 'to_tensor', 'train_single_forward',
           'single_device', 'TrainLoop', 'EvalLoop']

# Cell
from .loop import Loop,ProgressBar,Tolerate,Event,Stuff,chunkify
from types import MethodType
import numpy as np

# Cell
def create_event(event_name):
    class BatchEvent(Event):pass
    BatchEvent.__name__ = event_name
    return BatchEvent

def events(*enames):
    return list(map(create_event,enames))

# Cell
class LoopStack(Loop):
    settings = []
    """
    A stack of loop
    """
    @classmethod
    def from_loops(cls,*loops):
        def init(self,iterable=[],name = None):
            name = name if name!=None else cls.__name__

            self.loops = dict()
            l = Loop(iterable)
            for L in loops:
                l = L(iterable = l)
            super().__init__(iterable = l,
                             name = name)

            for stuff in cls.settings:
                self.make_stuff(stuff)

        setattr(cls,"init",init)
        return cls

    @classmethod
    def new_setting(cls,*settings):
        cls.settings+=list(settings)

    def make_stuff(self,name):
        new_stuff = Stuff(name)
        setattr(self.core,name,new_stuff)
        setattr(self,name,new_stuff)

    def __repr__(self,):
        return f"LoopStack>:{self.name}\n\t"+\
            "\n\t".join(map(str,self.core.layers[:-1]))

# Cell
import torch
from torch import is_tensor

def train_callbacks(loop):
    @loop.on_DATA_PROCESS
    def opt_zero_grad(loop):
        loop.opt("zero_grad")()

    @loop.before_1st_FORWARD
    def switch_model_to_train(loop):
        loop.model("train")()

    @loop.BACKWARD.on
    def opt_step(loop):
        loop.loss("backward")()

    @loop.BACKWARD.on
    def opt_step(loop):
        loop.opt("step")()

def to_tensor(x):
    return torch.Tensor(x)

def train_single_forward(metric_func = []):
    def train_single_forward_cb(self):
        @self.on_DATA_PROCESS
        def set_xy(self):
            self.var.x,self.var.y = self.element
            self.var.x = to_tensor(self.var.x)
            self.var.y = to_tensor(self.var.y)

        @self.on_FORWARD
        def forward_pass(self):
            y_ = self.model()(self.var.x)
            self.var.y_ = y_.popitem()[1][:,0]

        @self.on_LOSS_CALC
        def calculate_loss(self):
            for loss_name,loss_val in self.loss_func()(self.var.y_,self.var.y).items():
                self.loss[loss_name] = loss_val

        @self.on_METRICS
        def calcualte_metrics(self):
            # calculate metrics
            with torch.no_grad():
                self.metric.cases.update(self.metric_func()(self.var.y_,self.var.y))

        @self.on_METRICS
        def to_item(self):
            # loop through metrics
            dt = self.metric.cases
            dt.update(self.loss.cases)
            dt = dict((k,v.item() if is_tensor(v) else v) \
                          for k,v in dt.items())
            self.results.append(dt)
            self.pgbar_data(dt)

    return train_single_forward_cb

def single_device(device):
    def single_device_callback(self):
        @on_DATA_PROCESS
        def var_to_device(self):
            self.var.update("to")(device)

        @before_1st_FORWARD
        def model_to_device(self):
            self.model.update("to")(device)

    return single_device

class TrainLoop(LoopStack):
    def __init__(self,data_iter,model=[],opt=[],loss_func=[],loss=[],hp=[],cuda=[],
                 callbacks = [train_callbacks,],tolerate=True):
        loops = [ProgressBar,]
        if tolerate:
            loops.append(Tolerate)
        loops+=list(events(*TRAIN_EVENTS))
        self.from_loops(*loops)
        self.new_setting("model","var",
                         "opt","loss_func","loss",
                         "hp","cuda","metric_func","metric")
        self.init(data_iter,)
        for cb in callbacks:
            print(f"assigning callback {cb}")
            cb(self)

        self.core.results = []

class EvalLoop(LoopStack):
    def __init__(self,data_iter,tolerate=True):
        loops = [ProgressBar,]
        if tolerate:
            loops.append(Tolerate)
        loops+=list(events(*EVAL_EVENTS))
        self.from_loops(*loops)
        self.new_setting("model","var",
                         "loss_func","loss",
                         "hp","cuda","metric_func","metric")
        self.init(data_iter,)

        @self.EVAL_FORWARD.downstream
        def torch_eval_wrap(self,func):
            with torch.no_grad():
                func()