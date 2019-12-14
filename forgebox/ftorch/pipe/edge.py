from . import Col_Edge,DF_Edge
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd

from itertools import chain
from collections import Counter

class FillNa_Edge(Col_Edge):
    def __init__(self, fill=0.):
        super().__init__("fillna")
        self.fill = fill

    def colpro(self, col):
        return col.fillna(self.fill)


class EngTok_Edge(Col_Edge):
    def __init__(self, tokenizer, max_len=None):
        super().__init__("En")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def colpro(self, c):
        return c.apply(lambda x: self.tokenizer.tokenize(x)[:self.max_len])

eng_twt_tk = EngTok_Edge(TweetTokenizer())

class CNTok(Col_Edge):
    def __init__(self):
        """
        cntok = CNTok()
        datanode = start_node|cntok*["col1","col2"]
        datanode.run()
        """
        super().__init__("chinese_tokenize")
        from jieba import cut
        self.cut = cut

    def colpro(self, col):
        col = col.apply(lambda x:list(self.cut(str(x))))
        return col

class MinMax(Col_Edge):
    def __init__(self, min_ = None, max_ = None):
        super().__init__("minmax")
        self.min_ = min_
        self.max_ = max_

    def colpro(self,col):
        col.values = np.clip(col.values,a_min = self.min_, a_max = self.max_)
        return col

class TrackVocab(Col_Edge):
    """
    Col_Edge
    input column should contain python list
    This edge will keep track a vocabulary pandas DataFrame
    tck_vcb = TrackVocab()
    tck_vcb.vocab is the accumulated vocabulary
    """
    def __init__(self):
        super().__init__("track vocab")
        self.vocab = pd.DataFrame({"token": [], "cnt": []})

    def colpro(self, col):
        lists = list(col)
        biglist = list(chain.from_iterable(lists))
        self.vocab = self.combine_vocab(self.build_vocab(biglist))
        return col

    def get_token_count_dict(self, full_tok):
        """count the token to a list"""
        return Counter(full_tok)

    def build_vocab(self, full_tok):
        ct_dict = self.get_token_count_dict(full_tok)
        tk, ct = list(ct_dict.keys()), list(ct_dict.values())

        return pd.DataFrame({"token": tk, "cnt": ct})

    def combine_vocab(self, new_vocab):
        combinedf = pd.concat([self.vocab, new_vocab]).groupby("token").sum().reset_index()
        return combinedf.sort_values(by="cnt", ascending=False).reset_index().rename(columns = {"index":"idx"})

    def save_vocab(self, json_url):
        self.vocab.to_json(json_url)


class SaveCSV(DF_Edge):
    """
    DataFrame Edge
    SaveCsv("/path/to/file.csv")
    """
    def __init__(self, csvpath, tocsv_conf={"sep": "\t", "header": False}):
        super().__init__("save to csv")
        self.csvpath = csvpath
        self.tocsv_conf = tocsv_conf

    def pro(self, df):
        df.to_csv(self.csvpath, mode="a", **self.tocsv_conf)
        return df

class SaveSQL(DF_Edge):
    """
    DataFrame Edge
    SaveSQL("table_name", con)
    """
    def __init__(self, table_name, con, tosql_conf={"index": False, "if_exists":"append"}):
        super().__init__("save to sql_table")
        self.table_name = table_name
        self.con = con
        self.tosql_conf = tosql_conf

    def pro(self, df):
        df.to_sql(self.table_name, con=self.con, **self.tosql_conf)
        return df