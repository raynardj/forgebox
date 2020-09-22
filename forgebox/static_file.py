# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/12_static_file.ipynb (unless otherwise specified).

__all__ = ['get_static', 'open_static']

# Cell
from pathlib import Path
def get_static()->Path:
    """
    return the absolute path of forgebox.static
    """
    import forgebox
    return Path(forgebox.__path__[0])/"static"

def open_static(relative_path:str)->str:
    file = get_static()/relative_path
    with open(file,"r") as f:
        return f.read()