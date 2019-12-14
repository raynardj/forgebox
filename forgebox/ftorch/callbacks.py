import pandas as pd
from datetime import datetime
from .utils import mars

def recorddf(record):
    return pd.DataFrame(record)

## The following are the callbacks for trainer

def print_all(record):
    df = recorddf(record)
    mars(df)


def stat(record):
    """
    Print out track statistics
    :param record: self.track[epoch] in trainer.run(self.epoch)
    :param dataset: train dataset or val_dataset
    """
    df = recorddf(record)
    start = datetime.strptime(list(df["ts"])[0], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    end = datetime.strptime(list(df["ts"])[-1], "%Y-%m-%d %H:%M:%S.%f").timestamp()

    des = df.describe().loc[["mean", "min", "max"], :]
    des["timestamp"] = [(end - start) / len(df), 0, end - start]
    mars(des)


