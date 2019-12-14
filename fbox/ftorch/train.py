import __main__ as main
from torch.utils.data import DataLoader
from collections import namedtuple

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
                 conn=None, modelName="model", tryName="try", callbacks=[], val_callbacks=[]):
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
        train_data = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers) if val_dataset else None
        train_len = len(train_data)
        val_len = len(val_data) if val_data else None
        super().__init__(train_data, train_len=train_len, val_data=val_data, val_len=val_len,
                         fg=fg, print_on=print_on, fields=fields,
                         is_log=is_log, conn=conn, modelName=modelName,
                         tryName=tryName, callbacks=callbacks, val_callbacks=val_callbacks
                         )
