from torch.utils.data import Dataset,DataLoader
import math,os
import torch
import pandas as pd
import numpy as np
from collections import Counter


class Empty(Dataset):
    def __init__(self, length):
        """
        empety dataset, spit out random number
        :param length: how long you want this dataset
        """
        self.length = length
        self.seq = np.random.rand(length, 2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.seq[idx]

class DF_Dataset(Dataset):
    def __init__(self ,df, bs ,prepro=lambda x :x.values ,shuffle=False, col=None):
        """
        df_dataset, a dataset for slicing pandas dataframe,instead of single indexing and collate
        Please use batch_size=1 for dataloader, and define the batch size here
        eg.
        ```
        ds = DF_Dataset(data_df,bs=1024)
        ```
        """
        #         super(DF_Dataset, self).__init__()

        if shuffle:
            print("shuffling")
            df = df.sample(frac=1.).reset_index().drop("index" ,axis=1)
            print("shuffled") # shuffle the data here

        if col:
            df = df[col]
        self.df = df
        self.prepro = prepro
        self.bs = bs

    def __len__(self):
        return math.ceil(len(self.df) / self.bs)

    def __getitem__(self, idx):
        start = idx * self.bs
        end = (idx + 1) * self.bs
        #         print(type(self.x_prepro(self.df[start:end])),type(self.y_prepro(self.df[start:end])))
        return self.prepro(self.df[start:end])

class Arr_Dataset(Dataset):
    def __init__(self, *args, bs):
        """
        arr_dataset, a dataset for slicing numpy array,instead of single indexing and collate
        Please use batch_size=1 for dataloader, and define the batch size here
        eg.
        ```
        ds = Arr_Dataset(arr_1,arr_2,arr_3,bs = 512)
        ```
        """
        super(Arr_Dataset, self).__init__()
        self.arrs = args
        self.bs = bs

    def __len__(self):
        return math.ceil(len(self.arrs[0]) / self.bs)

    def __getitem__(self, idx):
        start = idx * self.bs
        end = (idx + 1) * self.bs
        return tuple(self.arrs[i][start:end] for i in range(len(self.arrs)))


class Seq_Dataset(Dataset):
    """
    a mechanism for reading sequence data, the preparation stage of nlp learning
    """

    def __init__(self,
                 seqname, seq, seq_len=None, vocab_path=None,
                 bs=16, vocab_size=10000, build_vocab=False, sep_tok="<tok>", discard_side="right", element_type="int", fixlen=False):
        '''
        A pytorch dataset for sequence
        seq: enumerable sequence
        vocab_path: path to vocabulary json file
        threads:process number
        element_type: "int" or "float"
        '''
        self.seqname = seqname
        print(seqname, "sequence type:", type(seq))
        self.process_funcs = []
        self.seq = list(seq)
        self.seq_len = seq_len
        self.vocab_path = vocab_path
        self.vocab_size = int(vocab_size)

        self.bs = bs
        self.calc_bn()
        print(seqname, "sequence total_length type:", self.N)

        self.sep_tok = sep_tok if sep_tok != None else ""
        self.make_breaker()
        self.discard_side = discard_side
        if self.seq_len:
            self.make_discard()

        if vocab_path:
            if build_vocab == False and os.path.exists(vocab_path) == False:
                print("vocab path not found, building vocab path anyway")
                build_vocab = True
            if build_vocab:
                self.vocab = self.build_vocab(self.seq)
                self.vocab.to_json(self.vocab_path)
            else:
                self.vocab = pd.read_json(self.vocab_path)
            self.char2idx, self.idx2char = self.get_mapping(self.vocab)
            self.make_translate()
        if fixlen:
            self.process_batch = self.process_batch_fixlen
        self.element_type = element_type
        self.make_totorch()

    def __len__(self):
        return self.BN

    def calc_bn(self):
        """
        calculate batch numbers
        :return: batch numbers
        """
        self.N = len(self.seq)
        self.BN = math.ceil(self.N / self.bs)
        return self.BN

    def __getitem__(self, idx):
        start_ = self.bs * idx
        seq_crop = self.seq[start_:start_ + self.bs]
        return self.process_batch(seq_crop)

    def make_breaker(self):
        if self.sep_tok != "":
            def breaker(self, line):
                return str(line).split(self.sep_tok)
        else:
            def breaker(self, line):
                return list(str(line))
        self.breaker = breaker
        self.process_funcs.append(self.breaker)

    def make_discard(self):
        if self.discard_side == "right":
            def discard(self, seqlist):
                return seqlist[:self.seq_len]
        if self.discard_side == "left":
            def discard(self, seqlist):
                return seqlist[-self.seq_len:]
        self.discard = discard
        self.process_funcs.append(self.discard)

    def make_translate(self):
        def translate(self, x):
            return np.vectorize(self.mapfunc)(x)

        self.translate = translate
        self.process_funcs.append(self.translate)

    def make_totorch(self):
        if self.element_type == "int":
            def totorch(self, seq):
                return torch.LongTensor(np.array(seq).astype(np.int))
        if self.element_type == "float":
            def totorch(self, seq):
                return torch.FloatTensor(np.array(seq).astype(np.float32))
        self.totorch = totorch
        self.process_funcs.append(self.totorch)

    def process_batch(self, batch):
        return torch.nn.utils.rnn.pad_sequence(np.vectorize(self.seq2idx, otypes=[list])(batch), batch_first=True)

    def process_batch_fixlen(self, batch):
        return torch.nn.utils.rnn.pad_sequence(np.vectorize(self.seq2idx, otypes=[list])(batch+[self.dummy_input()]), batch_first=True)[:-1,...]

    def dummy_input(self,):
        dummy = self.sep_tok.join([" "]*self.seq_len)
        return dummy

    def seq2idx(self, x):
        for f in self.process_funcs: x = f(self, x)
        return x

    def get_mapping(self, vocab_df):
        char2idx = dict(zip(vocab_df["token"], vocab_df["idx"]))
        idx2char = dict(zip(vocab_df["idx"], vocab_df["token"]))
        return char2idx, idx2char

    def mapfunc(self, x):
        """
        from token to index number
        """
        try:
            return self.char2idx[x]
        except:
            return 1

    def get_token_count_dict(self, full_token):
        """count the token to a list"""
        return Counter(full_token)

    def get_full_token(self, list_of_tokens):
        """
        From a list of list of tokens, to a long list of tokens, duplicate tokens included
        """
        return self.breaker(self, self.sep_tok.join(list_of_tokens))

    def build_vocab(self, seq_list):
        ct_dict = self.get_token_count_dict(self.get_full_token(seq_list))
        ct_dict["SOS_TOKEN"] = 9e9
        ct_dict["EOS_TOKEN"] = 8e9
        ct_dict[" "] = 7e9
        tk, ct = list(ct_dict.keys()), list(ct_dict.values())

        token_df = pd.DataFrame({"token": tk, "count": ct}).sort_values(by="count", ascending=False)
        return token_df.reset_index().drop("index", axis=1).reset_index().rename(columns={"index": "idx"}).fillna("")[
               :self.vocab_size]


class fuse(Dataset):
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(d.__getitem__(idx) for d in self.datasets)

def collate(batch):
    return tuple(i[0] for i in zip(*batch))


class col_core:
    def __init__(self, col_name, save_dir=".matchbox/fields", debug=False):
        os.system("mkdir -p %s" % (save_dir))
        self.col_name = col_name
        self.debug = debug
        self.save_dir = save_dir
        if self.save_dir[-1] != "/": self.save_dir += "/"

        self.meta = dict()

    def save_meta(self):
        np.save(self.save_dir + str(self.col_name) + ".npy", self.meta)

    def set_meta(self, meta=None):
        """
        pass meta dict values to obj attributes
        """
        if meta == None: meta = self.meta
        for k, v in meta.items():
            if self.debug: print("setting:\t%s\t%s" % (k, v))
            setattr(self, k, v)

    def load_meta(self, path=None):
        if path == None:
            path = self.save_dir + str(self.col_name) + ".npy"
        self.meta = np.load(path).tolist()
        self.set_meta(self.meta)

        if self.meta["coltype"] == "tabulate": self.make_sub()  # make sub-objects out of meta

    def make_meta(self):
        for attr in self.make_meta_list:
            self.meta[attr] = getattr(self, attr)

    def check_dim(self, data):
        return pd.DataFrame(data, columns=self.dim_names)


class categorical(col_core):
    def __init__(self, col_name, save_dir=".matchbox/fields"):
        super(categorical, self).__init__(col_name, save_dir)
        self.coltype = "categorical"
        self.make_meta_list = ["col_name", "coltype", "idx2cate", "cate2idx", "width", "eye", "dim_names"]
        self.__call__ = self.prepro

    def build(self, pandas_s, max_=20):
        assert max_ > 1, "max should be bigger than 1"

        vcount = pd.DataFrame(pandas_s.value_counts())

        print(vcount)

        self.cate_full = list(vcount.index.tolist())
        self.cate_list = self.cate_full[:max_ - 1]

        # build dictionary
        self.idx2cate = dict((k, v) for k, v in enumerate(self.cate_list))
        self.idx2cate.update({len(self.cate_list): "_other"})

        self.cate2idx = dict((v, k) for k, v in self.idx2cate.items())
        self.eye = np.eye(len(self.cate2idx))

        self.width = len(self.cate2idx)

        self.dim_names = list("%s -> %s" % (self.col_name, k) for k in self.cate2idx.keys())
        self.make_meta()

    def trans2idx(self, cate):
        """Translate category to index
        """
        try:
            return self.cate2idx[cate]
        except:
            return self.cate2idx["_other"]

    def prepro_idx(self, pandas_s):
        return pandas_s.apply(self.trans2idx)

    def prepro(self, pandas_s, expand=False):
        return self.eye[self.prepro_idx(pandas_s).values.astype(int)]

    def __call__(self, pandas_s, expand=False):
        return self.eye[self.prepro_idx(pandas_s).values.astype(int)]

    def dataset(self, df, bs,shuffle):
        return DF_Dataset(df, prepro=self.prepro, bs=bs, shuffle=shuffle,col = self.col_name)


class categorical_idx(col_core):
    """
    preprocessor
    """
    def __init__(self, col_name, save_dir=".matchbox/fields"):
        """
        column name
        :param col_name: str,column name
        :param save_dir: str, the place to save metadata, default .matchbox/fields
        """
        super(categorical_idx, self).__init__(col_name, save_dir)
        self.coltype = "categorical_idx"
        self.dim_names = [self.col_name]
        self.width = 1
        self.make_meta_list = ["col_name", "coltype", "idx2cate", "cate2idx", "width", "dim_names"]
        self.__call__ = self.prepro

    def build(self, pandas_s, max_=20):
        assert max_ > 1, "max should be bigger than 1"

        vcount = pd.DataFrame(pandas_s.value_counts())

        print(vcount)

        self.cate_full = list(vcount.index.tolist())
        self.cate_list = self.cate_full[:max_ - 1]

        # build dictionary
        self.idx2cate = dict((k, v) for k, v in enumerate(self.cate_list))
        self.idx2cate.update({len(self.cate_list): "_other"})

        self.cate2idx = dict((v, k) for k, v in self.idx2cate.items())

        self.make_meta()

    def trans2idx(self, cate):
        try:
            return self.cate2idx[cate]
        except:
            return self.cate2idx["_other"]

    def prepro(self, pandas_s, expand=True):
        x = pandas_s.apply(self.trans2idx).values
        if expand: x = np.expand_dims(x, -1)
        return x

    def __call__(self, pandas_s, expand=True):
        return self.prepro(pandas_s,expand)

    def dataset(self, df, bs,shuffle):
        return DF_Dataset(df, prepro=self.prepro, bs=bs, shuffle=shuffle,col = self.col_name)


class minmax(col_core):
    def __init__(self, col_name, fillna=0.0, save_dir=".matchbox/fields"):
        """minmax scaler: scale to 0~1"""
        super(minmax, self).__init__(col_name, save_dir)
        self.coltype = "minmax"
        self.fillna = fillna
        self.dim_names = [self.col_name]
        self.width = 1
        self.make_meta_list = ["col_name", "coltype", "min_", "max_", "range", "width", "dim_names"]
        self.__call__ = self.prepro

    def build(self, pandas_s=None, min_=None, max_=None):
        if type(pandas_s) != pd.core.series.Series:
            assert (min_ != None) and (max_ != None), "If no pandas series is set you have to set min_,max_ value"
            self.min_ = min_
            self.max_ = max_

        else:
            pandas_s = pandas_s.fillna(self.fillna)
            if min_ == None:
                self.min_ = pandas_s.min()
            else:
                self.min_ = min_
            if max_ == None:
                self.max_ = pandas_s.max()
            else:
                self.max_ = max_

        self.range = self.max_ - self.min_
        assert self.range != 0, "the value range is 0"
        print("min_:%.3f \tmax_:%.3f\t range:%.3f" % (self.min_, self.max_, self.range))
        self.make_meta()

    def prepro(self, data, expand=True):
        x = (np.clip(data.values.astype(np.float64), self.min_, self.max_) - self.min_) / self.range
        if expand: x = np.expand_dims(x, -1)
        return x

    def __call__(self, data,expand=True):
        return self.prepro(data,expand)

    def dataset(self, df, bs,shuffle):
        return DF_Dataset(df, prepro=self.prepro, bs=bs, shuffle=shuffle,col = self.col_name)

class tabulate(col_core):
    def __init__(self, table_name, save_dir=".matchbox/fields"):
        super(tabulate, self).__init__(table_name, save_dir)
        self.coltype = "tabulate"
        self.cols = dict()

        self.save_dir = save_dir
        if self.save_dir[-1] != "/":
            self.save_dir = "%s/" % (self.save_dir)

        self.make_meta_list = ["col_name", "coltype", "cols", "dim_names"]

    def build_url(self, metalist):
        for url in metalist:
            meta_dict = np.load(url).tolist()
            self.cols[meta_dict["col_name"]] = meta_dict
        self.make_dim()
        self.make_meta()

    def build(self, *args):
        for obj in args:
            self.cols[obj.col_name] = obj.meta
        self.make_sub()
        self.make_dim()
        self.make_meta()

    def make_col(self, meta):
        """
        creat sub obj according to sub meta
        """
        col_name = meta["col_name"]

        setattr(self, col_name, eval(meta["coltype"])(col_name))
        getattr(self, col_name).set_meta(meta)
        if meta["coltype"] == "tabulate":
            getattr(self, col_name).make_sub()
            getattr(self, col_name).meta = meta

    def make_sub(self):
        """
        create sub-objects according to meta
        """
        for k, meta in self.cols.items():
            self.make_col(meta)

    def make_dim(self):
        self.dim_names = []

        for k, meta in self.cols.items():
            for sub_dim in meta["dim_names"]:
                self.dim_names.append("%s -> %s" % (self.col_name, sub_dim))

        self.width = len(self.dim_names)

    def prepro(self, data):
        """
        data being a pandas dataframe
        """
        data_list = []

        for k, v in self.meta["cols"].items():
            # preprocess the data for every column
            col = getattr(self, k)
            if v["coltype"] == "tabulate":
                data_list.append(col.prepro(data))
            else:
                data_list.append(col.prepro(data[k]))
        return np.concatenate(data_list, axis=1)

    def __call__(self, data):
        return self.prepro(data)

    def dataset(self, df, bs,shuffle):
        """
        Get the dataset from the preprocess unit
        :param df: dataframe
        :param bs: batch size
        :param shuffle: if shuffle for the dataset
        :return: A pytorch dataset
        """
        return DF_Dataset(df, prepro=self.prepro, bs=bs, shuffle=shuffle)


class mapper:
    def __init__(self, conf_path, original=None, old_default=None, new_default=None, rank_size=None):
        """
        Handling mapping mechanism, all index mapping should be saved as config file
        [kwargs]

        conf_path: path of the configuration file, end with npy
        original: a list, will remove the duplicates
        old_default: defualt original value if the interpretaion failed
        new_default: defualt new value if the interpretaion failed

        user_map = mapper("conf/user_map.npy", user_Id_List )
        user_map.o2n will be the dictionary for original index to the new index
        user_map.n2o will be the dictionary for new index to the original index

        user_map = mappper("conf/user_map.npy") to load the conf file
        """
        self.conf_path = conf_path  # config file path
        self.old_default = old_default
        self.new_default = new_default
        self.rank_size = rank_size
        if original:
            self.original = list(set(original))
            self.mapping()
        else:
            self.load_map()

    def mapping(self):
        self.n2o = dict((k, v) for k, v in enumerate(self.original))
        self.o2n = dict((v, k) for k, v in enumerate(self.original))
        conf_dict = {"n2o": self.n2o, "o2n": self.o2n}
        if self.rank_size:
            chunk_size = int(len(self.original) / self.rank_size)
            last_size = len(self.original) - (self.rank_size - 1) * chunk_size
            conf_dict.update({"rank_len": [chunk_size] * (self.rank_size - 1) + [last_size]})
            conf_dict.update({"rank_n2o": list(
                dict(enumerate(self.original[chunk_size * i:chunk_size * i + conf_dict["rank_len"][i]])) for i in
                range(self.rank_size))})
            conf_dict.update({"rank_o2n": list(dict((v, k) for k, v in i.items()) for i in conf_dict["rank_n2o"])})
        np.save(self.conf_path, conf_dict)

    def load_map(self):
        dicts = np.load(self.conf_path).tolist()
        self.n2o = dicts["n2o"]
        self.o2n = dicts["o2n"]
        if "rank_len" in dicts:
            self.rank_len = dicts["rank_len"]
            self.rank_n2o = dicts["rank_n2o"]
            self.rank_o2n = dicts["rank_o2n"]

    def spit_new(self, o_idx):
        try:
            return self.o2n[o_idx]
        except:
            return self.new_default

    def spit_old(self, n_idx):
        try:
            return self.n2o[n_idx]
        except:
            return self.old_default

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
    ensure_factor, ensuring the row number to be the multiplication of this factor, default 2
    return train_df, valid_df
    """
    split_ = (np.random.rand(len(df)) > valid)
    train_df = df[split_].sample(frac=1.).reset_index().drop("index", axis=1)
    valid_df = df[~split_].sample(frac=1.).reset_index().drop("index", axis=1)

    if ensure_factor:
        train_mod = len(train_df) % ensure_factor
        valid_mod = len(valid_df) % ensure_factor
        if train_mod: train_df = train_df[:-train_mod]
        if valid_mod: valid_df = valid_df[:-valid_mod]
    return train_df, valid_df


class npNormalize(object):
    """
    normalize and denormalize for numpy
    """
    def __init__(self, v, mean=None, std=None):
        super().__init__()
        self.mean = v.mean() if mean == None else mean
        self.std = v.std() if std == None else std

    def normalize(self, x):
        return (x - self.mean) / self.std

    def recover(self, x):
        return (x * self.std) + self.mean


import random


class historyReplay(object):
    """
    A historic replay scheduler for GAN training

    ```python
    replay = historyReplay(bs = 32, # batch size
                current_ratio = 0.2 # for each batch keep 20% of the sample from the latest, default 0.2
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
                self.argslist = list(torch.cat([args[i], self.argslist[i]], dim=0) for i in range(len(self.argslist)))
                self.counter += 1
                return tuple(args) if self.arglen > 1 else tuple(args)[0]
            # above history length
            else:
                pos = self.counter % self.history_len
                start_pos = pos * self.bs
                end_pos = (pos + 1) * self.bs
                slice_ = random.choices(range(self.bs * self.history_len), k=self.history_chunk)
                rt = []
                for i in range(len(self.argslist)):
                    rt.append(torch.cat([args[i][:self.latest_chunk, ...], self.argslist[i][slice_, ...]], dim=0))
                    self.argslist[i][start_pos:end_pos, ...] = args[i]
                self.counter += 1
                return tuple(rt) if self.arglen > 1 else tuple(rt)[0]