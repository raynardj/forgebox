# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_loopstack.ipynb (unless otherwise specified).

__all__ = ['create_event', 'events', 'LoopStack', 'MetricTab', 'train_callbacks', 'eval_callbacks', 'to_tensor',
           'simple_forward', 'single_device', 'TrainLoop', 'EvalLoop', 'find_stuff', 'share_stuff']

# Cell
from .loop import Loop,ProgressBar,Tolerate,Event,Stuff,chunkify
from types import MethodType
from datetime import datetime
import numpy as np
import pandas as pd
from time import sleep

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
            for L in loops: l = L(iterable = l)
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
class MetricTab:
    """
    An object handling metric running
    """
    def __init__(self):
        self.df_by_epoch = []
        self.mean_by_epoch = []
        self.refresh_list()

    def __add__(self,row):
        self.results.append(row)
        return self

    def summary(self):
        """The summary on metrics"""
        return pd.DataFrame(self.mean_by_epoch)

    def update(self,**batch):
        self.batch.append(batch)
        return self

    def result_df(self):
        return pd.DataFrame(self.results)

    def batch_df(self):
        return pd.DataFrame(self.batch)

    def refresh_list(self):
        self.batch = []
        self.results = []

    def calc_mean(self,result_df,batch_df):
        product_df = pd.DataFrame(dict((col,result_df[col]*batch_df["bs"]) for col in self.result_cols))
        batch_mean = dict()
        for col in self.result_cols:
            batch_mean[col] = product_df[col].sum()/batch_df["bs"].sum()
        batch_mean["epoch"] = len(self.mean_by_epoch)+1
        batch_mean["span"] = batch_df.ts.max()-batch_df.ts.min()
        return batch_mean

    def mark(self):
        result_df = self.result_df()
        batch_df = self.batch_df()

        self.result_cols = result_df.columns
        for col in self.result_cols:
            batch_df[col] = result_df[col]

        batch_mean = self.calc_mean(result_df,batch_df)

        self.df_by_epoch.append(batch_df)
        self.mean_by_epoch.append(batch_mean)

        self.refresh_list()

        return batch_df, batch_mean

# Cell
def train_callbacks(loop):
    """
    call backs allow optimizing model weights
    """
    loop.core.metric_tab = MetricTab()

    @loop.every_start_FORWARD
    def switch_model_to_train(loop): loop.model("train")()

    @loop.on_DATA_PROCESS
    def opt_zero_grad(loop):loop.opt("zero_grad")()

    @loop.on_BACKWARD
    def opt_move(loop):
        loop.loss("backward")()
        loop.opt("step")()

def eval_callbacks(loop):
    loop.core.metric_tab = MetricTab()
    @loop.on_DATA_PROCESS
    def switch_model_to_eval(loop): loop.model("eval")()

def to_tensor(x):
    return torch.Tensor(x)

def simple_forward(metric_func = []):
    def simple_forward_cb(self):
        @self.on_DATA_PROCESS
        def set_xy(self):
            self.var.clear()
            self.var.x,self.var.y = self.element
            self.var.apply(to_tensor,scope = ["x","y"])()

        @self.on_FORWARD
        def forward_pass(self):
            y_ = self.model()(self.var.x)
            self.var.y_ = y_.popitem()[1][:,0]

        @self.on_LOSS_CALC
        def calculate_loss(self):
            losses =  self.loss_func()(self.var.y_,self.var.y)
            self.loss.update(losses)

        @self.on_METRICS
        def calcualte_metrics(self):
            # calculate metrics
            with torch.no_grad():
                metrics = self.metric_func()(self.var.y_,self.var.y)
                self.metric.update(metrics)

        @self.on_METRICS
        def to_item(self):
            # loop through metrics
            self.metric.update(self.loss.cases)
            dt = dict((k,v.item() if is_tensor(v) else v) \
                          for k,v in self.metric.cases.items())
            self.metric_tab+=dt
            self.pgbar_data(dt)

        @self.on_METRICS
        def save_batch_row(self):
            self.metric_tab.update(
                bs = self.var.x.size(0),
                epoch = self.epoch,
                i = self.i,
                ts = datetime.now()
            )

        @self.every_end_METRICS
        def show_metric(self):
            self.metric_tab.mark()
            summary =self.metric_tab.summary()
            try:
                from IPython.display import clear_output
                clear_output()
                display(summary)
            except:
                print(print(summary))

    return simple_forward_cb

def single_device(device):
    def single_device_callback(self):
        @self.on_DATA_PROCESS
        def var_to_device(self):
            self.var.apply("to")(device)

        @self.before_1st_FORWARD
        def model_to_device(self):
            self.model.apply("to")(device)

    return single_device

# Cell
import torch
from torch import is_tensor

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

#     @classmethod
#     def from_config(cls,train_data,val_data,model,opt,)

class EvalLoop(LoopStack):
    def __init__(self,data_iter,callbacks,tolerate=False):
        loops = [ProgressBar,]
        if tolerate:
            loops.append(Tolerate)
        loops+=list(events(*EVAL_EVENTS))
        self.from_loops(*loops)
        self.new_setting("model","var",
                         "loss_func","loss",
                         "hp","cuda","metric_func","metric")
        self.init(data_iter,)

        for cb in callbacks:
            print(f"assigning callback {cb}")
            cb(self)

        @self.FORWARD.downstream
        def torch_eval_wrap(self,func):
            with torch.no_grad():
                func()

# Cell
def find_stuff(core):
    klist = list(filter(lambda k:hasattr(getattr(core,k),"_is_stuff"), vars(core).keys()))
    return dict((k,getattr(core,k)) for k in klist)

def share_stuff(loop_from,loop_to):
    stuff_dict = find_stuff(loop_from.core)
    for k,v in stuff_dict.items():
        setattr(loop_to.core,k,v)