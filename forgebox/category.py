# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/53_category.ipynb (unless otherwise specified).

__all__ = ['C2I', 'Category', 'TreeCategory']

# Cell
import pandas as pd
import numpy as np
from pathlib import Path
import json
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Iterable, Dict, List


class C2I:
    """
    Category to indices
    >>> c2i = C2I(
            ["class 1", "class 2", ..., "class n"],
            pad_mst=True,
            )
    >>> c2i[["class 2", "class 5"]]
    [0] array([2,3])

    If the indices you put in the slicing is a np.ndarray
        a verctorized function will be used
    """

    def __init__(
        self,
        arr: Iterable,
        pad_mst: bool = False,
    ):
        self.pad_mst = pad_mst
        self.pad = ["[MST]", ] if self.pad_mst else []
        self.dict = dict(
            (v, k) for k, v in enumerate(self.pad + list(arr)))
        self.get_int = self.get_get_int()
        self.get_int_ = np.vectorize(self.get_int)

    def get_get_int(self,):
        if self.pad_mst:
            def get_int(idx: str) -> int:
                if idx in self.dict:
                    return self.dict[idx]
                else:
                    return 0
        else:
            def get_int(idx: str) -> int:
                return self.dict[idx]
        return get_int

    def __repr__(self) -> str:
        return f"C2I:{self.__len__()} categories"

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, k: int):
        if type(k) in [np.ndarray, list]:
            # use vectorized function
            return self.get_int_(k)
        else:
            # use the original python function
            return self.get_int(k)


class Category:
    """
    Manage categorical translations
    c = Category(
            ["class 1", "class 2", ..., "class n"],
            pad_mst=True,)

    c.c2i[["class 3","class 6"]]
    c.i2c[[3, 2, 1]]
    """

    def __init__(
        self,
        arr: Iterable,
        pad_mst: bool = False
    ):
        self.pad_mst = pad_mst
        self.c2i = C2I(arr, pad_mst=pad_mst)
        self.i2c = np.array(self.c2i.pad+list(arr))

    def save(self, path: Path) -> None:
        """
        save category information to json file
        """
        with open(path, "w") as f:
            json.dump(self.i2c.tolist(), f)

    @classmethod
    def load(cls, path: Path):
        """
        load category information from a json file
        """
        with open(path, "r") as f:
            l = np.array(json.load(f))
        if l[0] == "[MST]":
            return cls(l[1:], pad_mst=True)
        else:
            return cls(l, pad_mst=False)

    def __len__(self):
        return len(self.i2c)

    def __repr__(self):
        return f"Category Manager with {self.__len__()}"


class TreeCategory(Category):
    """
    Manage categorical translations
    c = Category(
            ["class 1", "class 2", ..., "class n"],
            pad_mst=True,)

    c.c2i[["class 3","class 6"]]
    c.i2c[[3, 2, 1]]
    """

    def __init__(
        self,
        parent_map: Dict[str, str],
        pad_mst: bool = False
    ):
        self.parent_map = parent_map
        arr = np.array(list(self.parent_map.keys()))
        super().__init__(arr, pad_mst=pad_mst)
        self.ancestor_map = dict()
        for name in self.parent_map.keys():
            self.find_ancestor_map(name)

        self.get_depth_map()
        self.get_depth_map_array()

    def find_ancestor_map(
        self, name: str
    ) -> Dict[str, List[str]]:
        if name in self.ancestor_map:
            return self.ancestor_map[name]
        if name not in self.parent_map:
            return []
        else:
            result = [name, ]+self.find_ancestor_map(self.parent_map[name])
            self.ancestor_map[name] = result
            return result


    def tree_hot(self, name: str) -> np.array:
        """
        return tree hot encoding name according to category
        """
        target = np.zeros(len(self), dtype=int)
        target[self.c2i[self.ancestor_map[name]]]=1
        return target

    def get_depth_map(self) -> Dict[str, int]:
        self.depth_map = dict(
            (k, len(v)) for k,v in self.ancestor_map.items())
        return self.depth_map

    def get_depth_map_array(self) -> np.array:
        self.depth_map_array = np.vectorize(
            self.depth_map.get)(self.i2c)
        return self.depth_map_array

    def __repr__(self):
        return f"""Tree Category({len(self)}):\n\tself.tree_hot("name")\tself.ancestor_map\tself.depth_map_array"""