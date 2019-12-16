import pandas as pd
import numpy as np
import nltk

tweet_tk =nltk.tokenize.TweetTokenizer()


class Node(object):
    def __init__(self, df, verbose=1):
        super(Node,self).__init__()
        self.df = df
        self.verbose = verbose
        self.pipenames = list()
        self.pipe = list()

    def __str__(self):
        return "<forge pipeline node>"+"|".join(self.pipenames)

    def __or__(self, edge):
        """
        use it as:
        node|edge|edge
        :param process_step:
        :return:
        """
        self.pipe.append(edge)
        self.pipenames.append(edge.edge_name)
        return self

    def run(self):
        for pro_i in range(len(self.pipe)):
            if self.verbose > 0: print("[df edge]:%s" % self.pipenames[pro_i])
            pro = self.pipe[pro_i]
            self.df = pro.pro(self.df)
        return self.df


class chunkNode(Node):
    def __init__(self, df, verbose=1):
        super().__init__(df, verbose)

    def run(self):
        """
        Running iterations on the entire dataset
        :return: None
        """
        for df in self.df:
            for pro_i in range(len(self.pipe)):
                if self.verbose > 0: print("[df edge]:%s" % self.pipenames[pro_i])
                pro = self.pipe[pro_i]
                df = pro.pro(df)

    def testrun(self):
        """
        testing for 1 iteration
        :return: the result dataframe
        """
        testdf = next(self.df)
        print("Please restart the generator after running test",flush=True)
        for pro_i in range(len(self.pipe)):
            if self.verbose > 0: print("[df edge]:%s" % self.pipenames[pro_i])
            pro = self.pipe[pro_i]
            testdf = pro.pro(testdf)

        return testdf


class frameEdge(object):
    def __init__(self, edge_name=None):
        super(frameEdge, self).__init__()
        if edge_name == None:
            edge_name = "DataFrame_Processing_Edge"
        self.edge_name = edge_name

    def __mul__(self, cols):
        assert 0, "Only Col_Edge support * columns operation"

    def define(self, f):
        def wraper(df):
            return f(df)

        self.pro = wraper
        return wraper


class colEdge(object):
    def __init__(self, edge_name=None):
        super().__init__()
        if edge_name == None:
            edge_name = "DataSeries_Processing_Edge"
        self.edge_name = edge_name
        self.cols = []

    def __mul__(self, cols):
        self.cols = cols
        return self

    def __mod__(self,col):
        self.cols = [col]
        return self

    def define(self, f):
        def wraper(col):
            col = f(col)
            return col

        self.colpro = wraper
        return wraper

    def pro(self, df):
        for c in self.cols:
            df[c] = self.colpro(df[c])
        return df

nothing = frameEdge("empety_process")

@nothing.define
def donothing(df):
    return df


