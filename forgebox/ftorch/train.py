from torch.utils.data import DataLoader
from collections import namedtuple
from types import MethodType
import torch
from forgebox.ftorch.optimizer import Opts
from forgebox.ftorch.utils import stty_size

from forgebox.utils import JUPYTER

TrainerBatch = namedtuple("TrainerBatch", ("epoch", "i", "data", "trainer"))
from forgebox.train import Trainer as Universal_Trainer
from .cuda import CudaHandler


class Trainer(Universal_Trainer):
    def __init__(self, dataset, val_dataset=None, batch_size=16, fg=None,
                 print_on=20, opts = [], batch_start_zg = True, fields=None, is_log=True, shuffle=True, num_workers=4,
                 conn=None, modelName="model", tryName="try", callbacks=[], val_callbacks=[], jupyter = JUPYTER, using_gpu = True):
        """
                Pytorch trainer
                fields: the fields you choose to print out
                is_log: writing a log？

                Training:

                write action function for a step of training,
                assuming a generator will spit out tuple x,y,z in each:
                then pass the function to object

                t=Trainer(...)
                t.train(epochs = 30)

                @t.step_train
                def action(self):
                    x,y,z = self.data

                    self.opt.sgd.zero_grad()
                    self.opt.adam.zero_grad()

                    # model is a global variable, or many models if you like
                    y_ = model(x)
                    y2_ = model_2(z)

                    ...... more param updating details here

                    # metrics you want to check return as dictionary here
                    return {"loss":loss.item(),"acc":accuracy.item()}

                same work for validation:trainer.val_action = val_action
                train_data: torch.utils.data.dataset.Dataset
                val_data: torch.utils.data.dataset.Dataset
                print_on: int, print metrics on progress for each n steps, default 20
                batch_size: int, default 16
                batch_start_zg: bool, zero grad all the optimizers at start, default True

                conn: a sql table connection, (sqlalchemy). if assigned value, save the record in the designated sql database;
                """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        train_data = self.ds_to_dl(dataset)
        val = (len(val_dataset)>0)
        val_data = self.ds_to_dl(val_dataset) if val else None
        train_len = len(train_data)
        val_len = len(val_data) if val else None
        self.before_train_batch_list = []
        self.before_val_batch_list = []
        super().__init__(train_data, train_len=train_len, val_data=val_data, val_len=val_len,
                         fg=fg, print_on=print_on, fields=fields,batch_start_zg = batch_start_zg,
                         is_log=is_log, conn=conn, modelName=modelName,
                         tryName=tryName, callbacks=callbacks, val_callbacks=val_callbacks,
                         jupyter = jupyter
                         )
        self.initialize_gpu(using_gpu)
        self.initialize_opt(opts)

    def ds_to_dl(self,ds):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_action_wrap(self):
        with torch.no_grad():
            return self.val_action()

    def before_train_batch(self, func_name):
        """
        @t.before_train_batch("process_data1")
        def pd1(self):
            ....
        """
        def wrapper(f):
            setattr(self,func_name, MethodType(f, self))
            self.before_train_batch_list.append(getattr(self,func_name))
        return wrapper

    def before_val_batch(self, func_name):
        """
        @t.before_val_batch("process_data1")
        def pd1(self):
            ....
        """
        def wrapper(f):
            setattr(self, func_name, MethodType(f, self))
            self.before_val_batch_list.append(getattr(self,func_name))
        return wrapper

    def data_to_dev(self,device ="cuda:0"):
        def to_dev(self):
            self.data = tuple(dt.to(device) for dt in self.data)
        self.step_extra("data_to_device")(to_dev)
        self.step_extra("val_data_to_device")(to_dev)

    def initialize_opt(self,opts):
        if type(opts)==list:
            self.opt = Opts(*opts)
            if len(opts) == 0:
                print(self.warning_no_optimizer)
        elif type(opts) == dict:
            self.opt = Opts(**opts)
        else:
            self.opt = Opts(opts)

    def initialize_gpu(self,using_gpu):
        """
        decide whether to use cuda device
        """
        self.using_gpu = False
        if (using_gpu == True) and (torch.cuda.is_available()):
            self.using_gpu = True
        if self.using_gpu:
            self.data_to_dev()

    @property
    def warning_no_optimizer(self):
        return f"""{stty_size()*"="}\nNotice, The Trainer was not initiated with optimizer
            Use the following syntax to initialize optimizer
            t.opt["adm1"] = torch.optim.Adam(m1.parameters())
            t.opt["adg1"] = torch.optim.Adagrad(m2.parameters())\n{stty_size()*"="}
            """
