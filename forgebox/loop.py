# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10_loop.ipynb (unless otherwise specified).

__all__ = ['Stuff', 'method4all', 'StorageCore', 'Loop', 'ProgressBar', 'error_tolerate', 'Tolerate', 'LambdaCall',
           'Event', 'chunkify']

# Cell
import numpy as np
import math

# Cell
class Stuff:
    def __init__(self,name):
        self.name = name
        self.cases = dict()
        self.funcs = dict()

    def __setitem__(self,k,v):
        self.cases[k]=v
        setattr(self,k,self.cases[k])

    def __getitem__(self,k):
        return self.cases[k]

    def append(self,obj):
        self[f"{self.name}_{len(self+1)}"] = obj

    def __getattr__(self,k):
        if k in self.cases:
            return self.cases[k]
        elif k in self.funcs:
            return self.funcs[k]
        else:
            return None

    def __repr__(self,):
        return f"⚽️:{self.name}[\n\t"+"\n\t".join(self.cases.keys())+"\n]"

    def __len__(self):
        return len(self.cases)

    def apply(self,f,name = None):
        """
        decorator
        append a function apply to all cases
        """
        def wraper(*args,**kwargs):
            for k,v in self.cases.items():
                f(v,*args,**kwargs)

        self.funcs[f.__name__] = wraper
        setattr(self,f.__name__,wraper)
        return wraper

    def __call__(self,*args,**kwargs):
        for k,func in self.funcs.items():
            func(*args,**kwargs)

# Cell
from tqdm import tqdm
from types import MethodType
import inspect
from functools import partial

def method4all(f):
    """
    Use this function as a decorator,
    The decorated function under Loop class can be used on outer layer
    """
    setattr(f,"forall",True)
    return f

class StorageCore:
    def __init__(self,start_layer):
        self.layers = []
        self.lmap = dict()
        self.forall_pool = dict()
        self.new_layer(start_layer)
        self.i = -1
        self.epoch = -1

    def new_layer(self,layer):
        layer.core = self
        self.layers.append(layer)

        self.lmap[layer._loop_layer]=layer
        if hasattr(self,layer.name):
            raise KeyError(f"Layer name already exist: {layer.name}")
        setattr(self,layer.name,layer)
        self.update_forall(layer)
        self.assign_forall(layer)

    def __repr__(self):
        return str(self.lmap)

    def for_all_functions(self,obj):
        return dict(inspect.getmembers(obj,
                  predicate=lambda x:hasattr(x,"forall")))

    def update_forall(self,obj):
        self.forall_pool.update(self.for_all_functions(obj))

    def assign_forall(self,obj):
        for name,f in self.forall_pool.items():
            setattr(obj,name,f)

class Loop:
    """
    Basic loop class
    """
    _loop_layer = 0.
    def __init__(self,iterable = [],name=None):
        self.iterable  = iterable
        self.next_layer = None
        self.name = name if name!=None else self.__class__.__name__

        if hasattr(iterable,"_loop_layer"):
            self._loop_layer = iterable._loop_layer + 1
            iterable.next_layer = self
            iterable.core.new_layer(self)
        else:
            self.core = StorageCore(self)

    def __call__(self):
        pass

    def __repr__(self):
        return f"layer🍰{self.name}"

    def on(self,iterable):
        self.iterable = iterable
        self._loop_layer = iterable._loop_layer + 1

    def func_detail(self,f):
        detail = dict({"🐍func_name":f.__name__,
                       "⛰doc":f.__doc__,
                       "😝var":",".join(f.__code__.co_varnames),
                       "😜names":",".join(f.__code__.co_names),
                      })
        return detail

    def summary(self):
        rt = f"Loop layer {self.name} summary:\n"
        rt+= "="*50+"\n"
        funcs = []
        for idx,layer in self.core.lmap.items():
            rt+= f"🍰layer{idx}\t{str(layer)}\n"
            for fname,f in self.core.for_all_functions(layer).items():
                if id(f) not in funcs:
                    rt+="\t"
                    rt+="\n\t".join(f"[{k}]\t{v}" for k,v in self.func_detail(f).items())
                    rt+="\n\t"+"."*35+"\n"
                    funcs.append(id(f))
            rt+="-"*50+"\n"
        rt+= "="*50+"\n"
        print(rt)

    def __len__(self):
        return self.iterable.__len__()

    def run(self,):
        """
        Run through iteration
        run start_call for every layer on the start of iteration
            run __call__ for every layer for each iteration
        run end_call for every layer on the end of iteration
        """
        first = self.layers[0]
        self.refresh_i()
        first.start_callon()
        for element in first:
            first()
        first.end_callon()

    def refresh_i(self):
        self.core.i=-1
        self.core.epoch+=1

    def update_i(self):
        self.core.i+=1

    def downstream_wrap(self,f):
        return f()

    def downstream_start_wrap(self,f):
        return f()

    def downstream_end_wrap(self,f):
        return f()

    def downstream(self,f):
        """
        set downstream wrapper on callback,
        The callback will happend in the local realm of the deocrated function
        example
        @loop.downstream
        def control_layer(self,callback):
            try:
                callback()
            except:
                print("error happened")
        """
        setattr(self,"downstream_wrap",MethodType(f,self))
        return f

    def downstream_start(self,f):
        """
        set downstream wrapper on start callback,
        The start_callback will happend in the local realm of the deocrated function
        example
        @loop.downstream_start
        def control_layer(self,start_callback):
            try:
                start_callback()
            except:
                print("error happened")
        """
        setattr(self,"downstream_start_wrap",MethodType(f,self))
        return f

    def downstream_end(self,f):
        """
        set downstream wrapper on end callback,
        The end_callback will happend in the local realm of the deocrated function
        @loop.downstream_end
        def control_layer(self,end_callback):
            try:
                end_callback()
            except:
                print("error happened")
        """
        setattr(self,"downstream_end_wrap",MethodType(f,self))
        return f

    def callon(self):
        self()
        self.downstream_wrap(self.iter_cb)

    def start_callon(self):
        self.start_call()
        self.downstream_start_wrap(self.start_cb)

    def end_callon(self):
        self.end_call()
        self.downstream_end_wrap(self.end_cb)

    def iter_cb(self):
        """
        call back during each iteration
        """
        if self.next_layer!=None:
            self.next_layer.callon()
        self.after()

    def after(self):
        pass

    def start_cb(self):
        """
        callback at the start of iteration
        """
        if self.next_layer!=None:
            self.next_layer.start_callon()

    def end_cb(self):
        """
        callback at the end of iteration
        """
        if self.next_layer!=None:
            self.next_layer.end_callon()

    def start_call(self):
        pass

    def end_call(self):
        pass

    def __iter__(self,):
        for element in self.iterable:
            if self._loop_layer ==0:
                self.update_i()
            self.core.element = element
            self.callon()
            yield self.element

    def __getattr__(self,k):
        return getattr(self.core,k)

    def is_newloop(self):
        """
        return Bool:Is this a new loop ready to start
        """
        return (self.i==-1 or self.i==self.__len__()-1)

# Cell
class ProgressBar(Loop):
    def __init__(self,iterable=[],jupyter = True,mininterval = 1e-1):
        super().__init__(iterable,"ProgressBar")

        if jupyter: # jupyter widget
            from tqdm.notebook import tqdm

        else: # compatible for console print
            from tqdm import tqdm

        self.tqdm = tqdm
        self.mininterval = mininterval
        self.data = dict()

    @method4all
    def pgbar_data(self,data):
        """
        update progress bar with python dictionary
        data: python dictionary
        """
        self.t.set_postfix(data)

    @method4all
    def pgbar_description(self,text):
        """
        update progress bar prefix with text string
        """
        self.t.set_description_str(f"{text}")

    def start_call(self):
        self.create_bar()

    def end_call(self):
        self.t.close()

    def __call__(self):
        self.t.update(1)

    def create_bar(self):
        self.t = self.tqdm(total=len(self.iterable),
                           mininterval=self.mininterval)

# Cell
def error_tolerate(self,downstream_func):
    """
    downstream_func:Downstream function
    """
    try:
        downstream_func()
    except Exception as e:
        self.errors.append(dict(stage=downstream_func.__name__,
                                i=self.i,
                                epoch=self.epoch,
                                error=e))

class Tolerate(Loop):
    """
    Tolerate any error happened downstream
    layer2 = Tolerate(upstream_iterable)
    # build downstream tasks
    layer3 = OtherApplication(layer2)
    layer3.run()
    # show the happened error message
    layer3.error_list()
    """
    def __init__(self,iterable = []):
        super().__init__(iterable,)
        self.errors = list()
        # wrap downstream task with error tolerate
        for decorator in [self.downstream,
                          self.downstream_start,
                          self.downstream_end]:
            decorator(error_tolerate)

    @method4all
    def error_list(self):
        """
        A list of errors happend so far
        """
        return self.errors

    def end_call(self,):
        le = len(self.error_list())
        if le>0:
            print(f"WARNING:{le} errors")

class LambdaCall(Loop):
    def __init__(self,iterable = [],
                 func = lambda x:x,
                 start_func = lambda x:x,
                 end_func = lambda x:x
                ):
        super().__init__(iterable,name=f"Lambda<{hex(id(self))}>")
        self.func = func
        self.start_func = end_func
        self.end_func = end_func

    def __call__(self):
        self.func(self)

    def start_call(self):
        self.start_func(self)

    def end_call(self):
        self.end_func(self)

# Cell

class Event(Loop):
    """
    An event is the landmark with in 1 iteration
    """
    def __init__(self,iterable=[]):
        event_name = self.__class__.__name__
        super().__init__(iterable,event_name)
        self.event_name = event_name
        self.cbs = []
        self.create_cb_deco("on")
        self.create_cb_deco("before_1st")
        self.create_cb_deco("after_last")
        self.core.update_forall(self)
        self.core.assign_forall(self)

    def __repr__(self):
        return f"event🌏{self.event_name}"

    def create_cb_deco(self,moment):
        event_name = self.event_name
        def wraper(cls,f):return getattr(self,moment)(f)
        wraper.__name__ = f"{moment}_{event_name}"
        wraper.__doc__ = f"""
            Append new {moment} callback for event:{event_name}
            Use this function as decorator
        """
        setattr(self,wraper.__name__,MethodType(method4all(wraper),self))

    def __call__(self):
        if len(self.cbs)>0:
            self.cbs[0].callon()

    def start_call(self):
        if len(self.cbs)>0:
            self.cbs[0].start_callon()

    def end_call(self):
        if len(self.cbs)>0:
            self.cbs[0].end_callon()

    def on(self,f):
        class EventCB(Loop):
            def __init__(self_,iterable=[]):
                super().__init__(iterable=iterable,
                                 name = f"ON_{self.event_name}_{self.cbs.__len__()+1}_{f.__name__}")
                self_.f = f

            def __call__(self_): self_.f(self)

        new_cb = EventCB()
        self.new_event_cb(new_cb)
        return f

    def before_1st(self,f):
        class EventCbBefore(Loop):
            def __init__(self_,iterable=[]):
                super().__init__(iterable=iterable,
                                 name = f"BEFORE_1ST_{self.event_name}_{self.cbs.__len__()+1}_{f.__name__}")
                self_.f = f

            def start_call(self_): self_.f(self)

        new_cb = EventCbBefore()
        self.new_event_cb(new_cb)
        return f

    def after_last(self,f):
        class EventCbAfter(Loop):
            def __init__(self_,iterable=[]):
                super().__init__(iterable=iterable,
                                 name = f"AFTER_LAST_{self.event_name}_{self.cbs.__len__()+1}_{f.__name__}")
                self_.f = f

            def end_call(self_): self_.f(self)

        new_cb = EventCbAfter()
        self.new_event_cb(new_cb)
        return f

    def new_event_cb(self,new_cb):
        new_cb.core.update_forall(new_cb)
        new_cb.core.assign_forall(new_cb)
        new_cb.core = self.core
        if len(self.cbs)>0:
            self.cbs[-1].next_layer = new_cb
#             self.cbs[-1].new_layer(new_cb)
        self.cbs.append(new_cb)

# Cell
def chunkify(*iterables,bs = 32):
    if len(iterables)>1:
        return list(tuple(l[i*bs:(i+1)*bs] for l in iterables) \
                for i in range(math.ceil(len(iterables[0])/bs)))
    else:
        return list(iterables[0][i*bs:(i+1)*bs] \
                for i in range(math.ceil(len(iterables[0])/bs)))