
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from typing import Callable


class ZipDataset(Dataset):
    """
    ZipDataset the dataset into one
    """

    def __init__(self, *datasets):
        """
        A pytorch dataset combining the dataset
        :param datasets:
        """
        self.datasets = datasets
        bs_s = set(list(d.bs for d in self.datasets))
        length_s = set(list(len(d) for d in self.datasets))
        assert len(bs_s) == 1, "batch sized not matched"
        assert len(length_s) == 1, "dataset lenth not matched"
        self.bs = list(bs_s)[0]
        self.length = list(length_s)[0]

    def __repr__(self) -> str:
        return f"ZipDataset: {self.datasets}"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(d.__getitem__(idx) for d in self.datasets)


def collate(batch):
    return tuple(i[0] for i in zip(*batch))


class test_DS:
    def __init__(self, dataset, *args, **kwargs):
        """
        pytorch dataset
        dt = test_DS(your_dataset, **kwargs)
        kwargs are the key word args for dataloader
        dt() to return the sample
        :param dataset:
        """
        self.dl = DataLoader(dataset, **kwargs)
        self.iter = iter(self.dl)

    def __call__(self):
        """
        returns data with iterator
        :return:data
        """
        return next(self.iter)


def split_df(df, valid=0.2, ensure_factor=2):
    """
    df: dataframe
    valid: valid ratio, default 0.1
    ensure_factor, ensuring the row number
        to be the multiplication of this factor, default 2
    return train_df, valid_df
    """
    split_ = (np.random.rand(len(df)) > valid)
    train_df = df[split_].sample(frac=1.).reset_index().drop("index", axis=1)
    valid_df = df[~split_].sample(frac=1.).reset_index().drop("index", axis=1)

    if ensure_factor:
        train_mod = len(train_df) % ensure_factor
        valid_mod = len(valid_df) % ensure_factor
        if train_mod:
            train_df = train_df[:-train_mod]
        if valid_mod:
            valid_df = valid_df[:-valid_mod]
    return train_df, valid_df


class npNormalize(object):
    """
    normalize and denormalize for numpy
    """

    def __init__(self, v, mean=None, std=None):
        super().__init__()
        self.mean = v.mean() if mean is not None else mean
        self.std = v.std() if std is not None else std

    def normalize(self, x):
        return (x - self.mean) / self.std

    def recover(self, x):
        return (x * self.std) + self.mean


class HistoryReplay(object):
    """
    A historic replay scheduler for GAN training

    ```python
    replay = historyReplay(bs = 32, # batch size
    # for each batch keep 20 persent of the sample from the latest, default 0.2
                current_ratio = 0.2,
                history_len = 50, # how long is the replay length, default 50
    )
    for i in range(iters):
        ...
        mixed_a,mixed_b,mixed_c = replay(a,b,c)
        ...
    ```
    """

    def __init__(self, bs, current_ratio=.2, history_len=50):
        self.current_ratio = current_ratio
        self.counter = 0
        self.history_len = history_len
        self.bs = bs
        self.argslist = []
        self.arglen = len(self.argslist)
        self.latest_chunk = int(bs * current_ratio)
        self.history_chunk = bs - self.latest_chunk

    def __call__(self, *args):
        # The 1st input
        if self.arglen == 0:
            self.argslist = args
            self.arglen = len(self.argslist)
            return tuple(args) if self.arglen > 1 else tuple(args)[0]
        else:
            stack_size = self.argslist[0].size(0)
            # the 2nd ~ the history length
            if stack_size < self.bs * self.history_len:
                self.argslist = list(torch.cat(
                    [args[i], self.argslist[i]], dim=0)
                    for i in range(len(self.argslist)))
                self.counter += 1
                return tuple(args) if self.arglen > 1 else tuple(args)[0]
            # above history length
            else:
                pos = self.counter % self.history_len
                start_pos = pos * self.bs
                end_pos = (pos + 1) * self.bs
                slice_ = random.choices(
                    range(self.bs * self.history_len), k=self.history_chunk)
                rt = []
                for i in range(len(self.argslist)):
                    rt.append(torch.cat(
                        [args[i][:self.latest_chunk, ...],
                         self.argslist[i][slice_, ...]], dim=0))
                    self.argslist[i][start_pos:end_pos, ...] = args[i]
                self.counter += 1
                return tuple(rt) if self.arglen > 1 else tuple(rt)[0]


def layering(dataset_class, new_name):
    """
    Instead of creating a bunch of dataset
    just treat another dataset as extra layer of __getitem__ function

    Inputs:
        - dataset_class: Dataset class, or functions
            that are decorated with this function
        - new_name: a name for next layer class

    eg.

    @layering(SomeStringDataset, "tensorDataset")
    def tokenizing(x):
        return tokenizer(x, return_tensors='pt')['input_ids'][0]

    @layering(tokenizing, "guessNextDataset")
    def guess_next(x):
        return x[:-1], x[1:]

    some_string_data = SomeStringDataset()

    tokenized_data = some_string_data.next_layer()
    print(tokenized_data[3])

    guess_pair = tokenized_data.next_layer()
    print(guess_pair[3])
    """
    if hasattr(dataset_class, "associated_class"):
        dataset_class = dataset_class.associated_class

    def decorator(f: Callable):
        class newClass(Dataset):
            def __init__(self, last_ds,):
                self.last_ds = last_ds

            def __repr__(self):
                return f"Layered Pytorch Dataset: {new_name}"

            def __len__(self): return len(self.last_ds)

            def __getitem__(self, idx):
                return f(self.last_ds[idx])

        newClass.__name__ = new_name

        def next_layer(self,):
            return newClass(self)
        f.associated_class = newClass
        dataset_class.next_layer = next_layer
        return f
    return decorator
