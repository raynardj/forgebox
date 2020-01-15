import os
from datetime import datetime
import __main__ as main
from tqdm import trange
from functools import reduce
import pandas as pd
from collections import namedtuple


try:
    JUPYTER = True if main.get_ipython else False
except:
    JUPYTER = False

if JUPYTER: from tqdm import tqdm_notebook as tn

TrainerBatch = namedtuple("TrainerBatch", ("epoch", "i", "data", "trainer"))

class Trainer:
    def __init__(self, train_data, train_len, val_data=None, val_len=None,  batch_size=16, fg=None,
                 print_on=20, fields=None, is_log=True,
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

        self.callbacks = callbacks
        self.val_callbacks = val_callbacks

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


    def get_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    def train(self, epochs, name=None, log_addr=None):
        """
        Train the model for some epochs
        """
        if self.fg:
            self.fg.new_train()

        if name == None:
            name = "torch_train_" + datetime.now().strftime("%y%m%d_%H%M%S")
        if log_addr == None:
            log_addr = ".log_%s" % (name)

        if log_addr[-1] != "/": log_addr += "/"

        for epoch in range(epochs):
            self.track[epoch] = list()
            self.run(epoch)
        if self.is_log:
            os.system("mkdir -p %s" % (log_addr))
            trn_track = pd.DataFrame(reduce((lambda x, y: x + y), list(self.track.values())))
            trn_track.to_csv(log_addr + "trn_" + datetime.now().strftime("%y_%m_%d__%H_%M_%S") + ".csv",
                             index=False)

            if self.val_len:
                val_track = pd.DataFrame(reduce((lambda x, y: x + y), list(self.val_track.values())))
                val_track.to_csv(log_addr + "val_" + datetime.now().strftime("%y_%m_%d__%H_%M_%S") + ".csv",
                                 index=False)

    def run(self, epoch):
        """
        run for a single epoch
        :param epoch: the epoch index
        :return:
        """
        if JUPYTER:
            t = tn(range(self.train_len))
        else:
            t = trange(self.train_len)
        self.train_gen = iter(self.train_data)

        for i in t:
            batch = TrainerBatch(epoch, i, next(self.train_gen), self)
            ret = self.action(batch)
            ret.update({"epoch": epoch,
                        "iter": i,
                        "ts": self.get_time()})
            self.track[epoch].append(ret)

            if i % self.print_on == self.print_on - 1:
                self.update_descrition(epoch, i, t)
        for cb_func in self.callbacks:
            cb_func(record=self.track[epoch])

        if self.val_len:

            self.val_track[epoch] = list()
            self.val_gen = iter(self.val_data)
            if JUPYTER:
                val_t = tn(range(self.val_len))
            else:
                val_t = trange(self.val_len)

            for i in val_t:
                batch = TrainerBatch(epoch, i, next(self.val_gen), self)
                ret = self.val_action(batch)
                ret.update({"epoch": epoch,
                            "iter": i,
                            "ts": self.get_time()})
                self.val_track[epoch].append(ret)
                self.update_descrition_val(epoch, i, val_t)

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
        if JUPYTER:
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

    def step_train(self, f):

        """
        A decorator: @trainer.step_train, following the training step function
        :param f:
        :return:
        """

        def wraper(batch):
            return f(batch)

        self.action = wraper
        return wraper

    def step_val(self, f):
        """
        A decorator: @trainer.step_val, following the validation step function
        :param f:
        :return:
        """

        def wraper(batch):
            return f(batch)

        self.val_action = wraper
        return wraper
