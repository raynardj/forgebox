# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_loopstack.ipynb (unless otherwise specified).

__all__ = ['LoopStack', 'create_event', 'events']

# Cell
from .loop import Loop,ProgressBar,Tolerate,Event
from types import MethodType

# Cell
class LoopStack(Loop):
    @classmethod
    def from_loops(cls,*loops):
        cls.Loops = dict()
        for l in loops:
            cls.Loops.update({l.__name__:l})
            setattr(cls,l.__name__,l)
        def init(self,iterable=[],name = None):
            name = name if name!=None else cls.__name__
            super().__init__(iterable = iterable,
                             name = name)
            self.loops = dict()
            l = self
            for L in loops:
                l = L(l)
                setattr(self,L.__name__,l)
                self.loops.update({L.__name__:l})

        setattr(cls,"__init__",init)
        return cls

    def __repr__(self,):
        return f"LoopStack>:{self.name}\n\t"+\
            "\n\t".join(map(lambda x:"Layer>>:"+x,self.loops.keys()))

# Cell
def create_event(event_name):
    class BatchEvent(Event):pass
    BatchEvent.__name__ = event_name
    return BatchEvent

def events(*enames):
    return list(map(create_event,enames))