from ..apicore import forgedb
from .callbacks import recorddf
import torch
import pandas as pd
import os
from datetime import datetime

class FG(forgedb):
    def __init__(self, *args, **kwargs):
        super(FG, self).__init__(*args, **kwargs)

    # callbacks
    def metrics(self, adapt=["mean"], ):
        """
        return a function to save metrics
        :param adapt: list, default ["mean"], possible values:"mean","min","max","std","20%","50%","70%"
        """
        def func(record):
            df = recorddf(record)
            des = df.describe().loc[adapt, :]
            metric_dict = dict()

            epoch_now = list(df.epoch)[-1]
            des = des.drop("epoch", axis=1)
            des = des.drop("iter", axis=1)
            for col in des.columns:
                des.apply(lambda x: metric_dict.update({"%s_%s" % (x.name, col): x[col]}), axis=1)
            if self.verbose:
                print(metric_dict, flush=True)
            self.save_metrics(metrics=metric_dict, epoch = epoch_now)
            return metric_dict

        return func

    def weights(self, model, name=None):
        """
        A callback function to save weights
        fg = FG(task = "wgan")
        Trianer(...., callbacks = [fg.wegiths(model_G, name='wgan_g'), fg.weights(model_D,name = 'wgan_d')])
        :param model: A pytorch model
        :param name: Name string of the model, no space and strange charactors
        :return: a function, result of the decorator
        """
        fgobj = self
        name_ = name
        if name_ == None:
            name_ = self.new_model_name()
        else:  # todo: add a regex to validate a consequtive string
            name_ = "%s_%s" % (self.train.id, name_)

        def f(record):
            epoch = list(recorddf(record).epoch)[0]
            name_epoch = "%s.e%s" % (name_, epoch)
            path = self.weightdir / ("%s" % (name_epoch if name_epoch[-4:] == ".npy" else "%s.npy" % (name_epoch)))
            if fgobj.verbose: print("[Model Save]:%s" % (path))
            torch.save(model.state_dict(), path)
            return fgobj.save_weights(path, modelname=name_epoch, framewk="pytorch")

        return f

    def logs(self, train=True):
        """
        Saving the logs for training validation to csv
        :param train: Bool, True for training, False for validation
        :return: a function, result of the decorator
        """
        def f(record):
            df = recorddf(record)
            epoch = list(df.epoch)[0]
            path = self.logsdir / (
                        "%s_%s.%s.csv" % (self.task, self.train.id if train else "val.%s" % (self.train.id), epoch))
            path = str(path)
            df.to_csv(path, index=False)
            self.log_record(path)

        return f


class loopTracker(object):
    def __init__(self):
        """
        Initialize:
        ```
        l = loopTracker()
        ```
        usage:
        ```
        for e in range(epochs):
            for i in range(iters):
                x,y1,y2 = next(shuffled_data_gen)
                y1_, y2_ = model(x)
                # tracking data in loop like this
                l(e, i , y1, y2, y1_, y2_)
                ...

        tracked_data_df = l.df(cols = ["epoch","iter", "y1","y2","y1_pred","y2_pred"])
        ```
        """
        self.seq = list()

    def __call__(self, *args):
        self.seq.append(args)
        return tuple(args)

    def __len__(self):
        return len(self.seq)

    @property
    def width(self):
        if self.__len__() > 0:
            return len(self.seq[0])
        else:
            return 0

    def __repr__(self):
        return "<loopTracker:Tracked %s Rows, %s Cols>" % (self.__len__(), self.width)

    def df(self, cols=None):
        """
        cols:
        a list of str: name of columns
        if cols == None, the column name would be sequenced natural numbers
        """
        if cols == None:
            cols = list("col_%s" % (i + 1) for i in range(self.width))
        df = pd.DataFrame(self.seq, columns=cols)
        return df
