# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_freemap.ipynb (unless otherwise specified).

__all__ = ['FreeMap']

# Cell
from collections import OrderedDict

# Cell
class FreeMap:
    def __init__(self,
                 func= lambda x:x,
                 filter_function = lambda x: True,
                 flatten = False):
        """
        run map operation on a free stucture,
        like dictionary, list within dictionary, etc

        """
        self.func = func
        self.filter_function = filter_function
        self.flatten = flatten
        if flatten:
            self.result_list = []
        self.create_doc()

    def __repr__(self):
        return f"""<A map function for free structure>
        Please put in one of the following type:
        dict, list, tuple, set, OrderedDict

        {self.keep_structure}
        ======
        Mapped function
        {self.func}
        Value Filter function
        {self.filter_function}
        """

    @property
    def keep_structure(self):
        return "This operation will flatten the structure to a list"\
            if self.flatten else\
            "This operation will keep the original structure"

    def create_doc(self):
        doc = f"""
        map function for list,dict,tuple,set,OrderedDict
        {self.keep_structure}

        """
        if hasattr(self.func,"__doc__"):
            if self.func.__doc__!=None:
                doc += f"doc string from mapping function:\n\t\t{self.func.__doc__}\t"
        setattr(self,"__doc__",doc)

    def parse(self,thing):
        if type(thing) in [list,tuple,set]:
            return self.parse_seq(thing)
        elif type(thing) in [dict,OrderedDict]:
            return self.parse_kv(thing)
        else:
            return self.parse_value(thing)

    def parse_seq(self,thing):
        return type(thing)(self.parse(i) for i in thing)

    def parse_kv(self,thing):
        return type(thing)((k,self.parse(v)) for k,v in thing.items())

    def parse_value(self,thing):
        if self.filter_function(thing):
            if self.flatten==False:
                return self.func(thing)
            else:
                self.result_list.append(self.func(thing))
        return thing

    def __call__(self,iterable):
        result = self.parse(iterable)
        if self.flatten:
            return self.result_list
        else:
            return result