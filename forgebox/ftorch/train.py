import __main__ as main
from torch.utils.data import DataLoader
from collections import namedtuple
import torch

try:
    JUPYTER = True if main.get_ipython else False
except:
    JUPYTER = False

if JUPYTER: from tqdm import tqdm_notebook as tn

TrainerBatch = namedtuple("TrainerBatch", ("epoch", "i", "data", "trainer"))
from forgebox.train import Trainer as Universal_Trainer


class Trainer(Universal_Trainer):
    def __init__(self, dataset, val_dataset=None, batch_size=16, fg=None,
                 print_on=20, fields=None, is_log=True, shuffle=True, num_workers=4,
                 conn=None, modelName="model", tryName="try", callbacks=[], val_callbacks=[], jupyter = JUPYTER):
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

                    return {"loss":loss.data[0],"acc":accuracy.data[0]}

                same work for validation:trainer.val_action = val_action

                conn: a sql table connection, (sqlalchemy). if assigned value, save the record in the designated sql database;
                """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        train_data = self.ds_to_dl(dataset)
        val_data = self.ds_to_dl(val_dataset) if val_dataset else None
        train_len = len(train_data)
        val_len = len(val_data) if val_data else None
        super().__init__(train_data, train_len=train_len, val_data=val_data, val_len=val_len,
                         fg=fg, print_on=print_on, fields=fields,
                         is_log=is_log, conn=conn, modelName=modelName,
                         tryName=tryName, callbacks=callbacks, val_callbacks=val_callbacks,
                         jupyter = JUPYTER
                         )

    def ds_to_dl(self,ds):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def step_val(self, f):
        """
        A decorator: @trainer.step_val, following the validation step function
        The function will run under a torch.no_grad() session
        :param f: A function taking in the parameter: batch
        :return:
        """
        def wraper(batch):
            with torch.no_grad():
                return f(batch)

        self.val_action = wraper
        return wraper


