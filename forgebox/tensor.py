__all__ = ['modules_to_opt_conf','measure_freeze','numpify']

from copy import deepcopy
from torch import nn

def modules_to_opt_conf(*modules,**opt_kwargs)->"a list of PyTorch optimizer config":
    """
    put in a sequence of pytorch modules, 
    return optimizer param groups of configs
    """
    param_list = []
    param_list_nd = []
    no_decay = ["bias", "LayerNorm.weight"]

    for m in modules:
        for n,p in m.named_parameters():
            if any(nd in n for nd in no_decay):
                param_list_nd.append(p)
            else:
                param_list.append(p)
                
    opt_kwargs_nd = deepcopy(opt_kwargs)
    opt_kwargs_nd["weight_decay"]=0.

    return [dict(params=param_list,**opt_kwargs), # param_group with weight decay
            dict(params=param_list_nd,**opt_kwargs_nd), # param_group without weight decay
           ]

def measure_freeze(m:nn.Module)->"a describtion about how many submodules are unfreezed":
    """
    measure the how many sub-module we freezed or unfreezed
    """
    total = 0
    trainable = 0
    for param in m.parameters():
        total+=1
        if param.requires_grad: trainable+=1
    return f"{trainable} trainables/{total} total"

def numpify(*tensors):
    tensors = list(tensors)
    """Transform bunch of pytorch tensors to numpy array, even if in cuda"""
    for i in range(len(tensors)):
        tensors[i] = tensors[i].clone().cpu().numpy()
    return tensors