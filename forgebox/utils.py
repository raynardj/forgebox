import os
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tell_jupyter():
    import __main__ as main

    try:
        JUPYTER = True if main.get_ipython else False
    except:
        JUPYTER = False

    return JUPYTER

JUPYTER = tell_jupyter()