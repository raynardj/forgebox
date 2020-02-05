from forgebox.utils import JUPYTER
if JUPYTER: from IPython.display import display
import torch
import os

def mars(printstuff):
    """
    A print function works between jupyter and console print
    :param printstuff:
    """
    if JUPYTER:display(printstuff)
    else:print(printstuff)

def save_model(model,path):
    torch.save(model.state_dict(), path)

def load_model(model,path):
    model.load_state_dict(torch.load(path))

def stty_size():
    try:
        return int(os.popen('stty size', 'r').read().split()[1])
    except:
        return 60