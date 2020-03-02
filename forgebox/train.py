import os
from datetime import datetime
from tqdm import trange
from functools import reduce
import pandas as pd
from collections import namedtuple
from types import MethodType
from forgebox.utils import JUPYTER

if JUPYTER: from tqdm import tqdm_notebook as tn

TrainerBatch = namedtuple("TrainerBatch", ("epoch", "i", "data", "trainer"))

class Trainer:
    def __init__(self, train_data, train_len, val_data=None, val_len=None,  batch_size=16, fg=None,
                 print_on=20, fields=None, is_log=True,batch_start_zg = True,
                 conn=None, modelName="model", tryName="try", callbacks=[], val_callbacks=[],jupyter = JUPYTER):
        """
        A training iteration wraper
        fields: the fields you choose to print out
        is_log: writing a log？

        Training:

        write action function for a step of training,
        assuming a generator will spit out tuple x,y,z in each:
        then pass the function to object

        t=Trainer(...)
        t.train(epochs = 30)

        @t.step_train
        def action(batch):
            x,y,z = batch.data
            x,y,z = x.cuda(),y.cuda(),z.cuda()

            #optimizer is a global variable, or many different optimizers if you like
            sgd.zero_grad()
            adam.zero_grad()

            # model is a global variable, or many models if you like
            y_ = model(x)
            y2_ = model_2(z)

            ...... more param updating details here

            return {"loss":loss.item(),"acc":accuracy.item()}

        same work for validation:trainer.val_action = val_action

        conn: a sql table connection, (sqlalchemy). if assigned value, save the record in the designated sql database;
        """
        self.batch_size = batch_size
        self.conn = conn
        self.fg = fg
        self.modelName = modelName
        self.tryName = tryName
        self.train_data = train_data
        self.train_len = train_len
        self.val_data = val_data
        self.val_len = val_len
        self.print_on = print_on
        self.jupyter = jupyter

        self.callbacks = callbacks
        self.val_callbacks = val_callbacks
        self.batch_start_zg = batch_start_zg

        self.before_train_batch_list = []
        self.before_val_batch_list = []

        if self.val_data:
            self.val_track = dict()

        self.track = dict()
        self.fields = fields
        self.is_log = is_log

    def one_batch(self):
        if hasattr(self, "testgen") == False:
            self.testgen = iter(self.train_data)
        try:
            return next(self.testgen)
        except StopAsyncIteration:
            self.testgen = iter(self.train_data)
            return next(self.testgen)

    def progress(self, l):
        return tn(range(l)) if self.jupyter else trange(l)

    def get_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    def train(self, epochs, name=None, log_addr=None):
        """
        Train the model for some epochs
        """
        if self.fg:
            self.fg.new_train()

        for epoch in range(epochs):
            self.track[epoch] = list()
            self.run(epoch)
        self.log(name = name,log_addr=log_addr)


    def log(self, name, log_addr):
        if self.is_log:
            if name == None:
                name = "torch_train_" + datetime.now().strftime("%y%m%d_%H%M%S")
            if log_addr == None:
                log_addr = ".log_%s" % (name)

            if log_addr[-1] != "/": log_addr += "/"
            os.system("mkdir -p %s" % (log_addr))
            trn_track = pd.DataFrame(reduce((lambda x, y: x + y), list(self.track.values())))
            trn_track.to_csv(log_addr + "trn_" + datetime.now().strftime("%y_%m_%d__%H_%M_%S") + ".csv",
                             index=False)

            if self.val_len:
                val_track = pd.DataFrame(reduce((lambda x, y: x + y), list(self.val_track.values())))
                val_track.to_csv(log_addr + "val_" + datetime.now().strftime("%y_%m_%d__%H_%M_%S") + ".csv",
                                 index=False)

    def train_iteration(self, i, t):
        self.i = i
        self.data = next(self.train_gen)

        for f in self.before_train_batch_list: f()

        if self.batch_start_zg: self.opt.zero_all()

        if hasattr(self,"data_to_device"): self.data_to_device()

        ret = self.action()
        ret.update({"epoch": self.epoch,
                    "iter": i,
                    "ts": self.get_time()})
        self.track[self.epoch].append(ret)

        if i % self.print_on == self.print_on - 1:
            self.update_descrition(self.epoch, i, t)

    def val_action_wrap(self):
        return self.val_action()

    def val_iteration(self, i ,val_t):
        self.i = i
        self.data = next(self.val_gen)
        for f in self.before_val_batch_list: f()

        if hasattr(self, "val_data_to_device"): self.val_data_to_device()

        ret = self.val_action_wrap()
        ret.update({"epoch": self.epoch,
                    "iter": i,
                    "ts": self.get_time()})
        self.val_track[self.epoch].append(ret)
        self.update_descrition_val(self.epoch, i, val_t)

    def run(self, epoch):
        """
        run for a single epoch
        :param epoch: the epoch index
        :return:
        """
        t = self.progress(self.train_len)

        self.train_gen = iter(self.train_data)
        self.epoch = epoch

        for i in t:
            self.train_iteration(i,t)

        for cb_func in self.callbacks:
            cb_func(record=self.track[epoch])

        if self.val_len:
            self.epoch = epoch
            self.val_track[epoch] = list()
            self.val_gen = iter(self.val_data)

            val_t = self.progress(self.val_len)
            for i in val_t:
                self.val_iteration(i, val_t)

            for v_cb_func in self.val_callbacks:
                v_cb_func(record=self.val_track[epoch] )

    def update_descrition(self, epoch, i, t):
        window_df = pd.DataFrame(self.track[epoch][max(i - self.print_on, 0):i])

        if self.conn:  # if saving to a SQL database
            window_df["split_"] = "train"
            window_df["tryName"] = self.tryName + "_train"
            window_df.to_sql("track_%s" % (self.modelName), con=self.conn, if_exists="append", index=False)
        window_dict = dict(window_df.mean())
        del window_dict["epoch"]
        del window_dict["iter"]

        desc = "⭐[ep_%s_i_%s]" % (epoch, i)
        if self.jupyter:
            t.set_postfix(window_dict)
        else:
            if self.fields != None:
                desc += "✨".join(list("\t%s\t%.3f" % (k, v) for k, v in window_dict.items() if k in self.fields))
            else:
                desc += "✨".join(list("\t%s\t%.3f" % (k, v) for k, v in window_dict.items()))
        t.set_description(desc)

    def update_descrition_val(self, epoch, i, t):
        if self.conn:  # if saving to a SQL database
            window_df = pd.DataFrame(self.val_track[epoch][max(i - self.print_on, 0):i])
            window_df["split_"] = "valid"
            window_df["tryName"] = self.tryName + "_valid"
            window_df.to_sql("track_%s" % (self.modelName), con=self.conn, if_exists="append", index=False)
        window_dict = dict(pd.DataFrame(self.val_track[epoch]).mean())
        # print(pd.DataFrame(self.val_track[epoch]))
        del window_dict["epoch"]
        del window_dict["iter"]

        desc = "😎[val_ep_%s_i_%s]" % (epoch, i)
        if JUPYTER:
            t.set_postfix(window_dict)
        else:
            if self.fields != None:
                desc += "😂".join(list("\t%s\t%.3f" % (k, v) for k, v in window_dict.items() if k in self.fields))
            else:
                desc += "😂".join(list("\t%s\t%.3f" % (k, v) for k, v in window_dict.items()))
        t.set_description(desc)

    def todataframe(self, dict_):
        """return a dataframe on the train log dictionary"""
        tracks = []
        for i in range(len(dict_)):
            tracks += dict_[i]

        return pd.DataFrame(tracks)

    def step_train(self,f):
        setattr(self, "action", MethodType(f, self))
        return f

    def step_val(self,f):
        setattr(self, "val_action", MethodType(f, self))
        return f

    def step_extra(self, func_name):
        """
        @t.step_extra("forward_pass")
        def fp(self):
            self.x = self.data.x.float()
            self.y = self.datay.long()
            self.y_ = model(self.x)
        """
        def assign(f):
            setattr(self,func_name, MethodType(f, self))
        return assign

