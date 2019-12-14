from .train import JUPYTER
if JUPYTER: from IPython.display import display
import torch

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