from category import Category
import numpy as np

def cache(f):
    data = dict()
    def wrapper(name, parent_map):
        if name in data:
            return data[name]
        rt = f(name, parent_map)
        data[name]=rt
        return rt
    return wrapper

@cache
def find_ancestor_map(name, parent_map):
    if name not in parent_map:
        return []
    else:
        return [name,]+find_ancestor_map(parent_map[name], parent_map)

def tree_hot(cate, name, ancestor_map):
    target = np.zeros(len(cate), dtype=int)
    target[cate.c2i[ancestor_map[name]]]=1
    return target

def get_depth_map(cate, ancestor_map):
    cate.depth_map = dict(
        (k, len(v)) for k,v in ancestor_map.items())
    return cate.depth_map

def get_depth_map_array(cate, ancestor_map):
    cate.depth_map_array = np.vectorize(cate.get_depth_map(ancestor_map).get)(cate.i2c)
    return cate.depth_map_array

Category.tree_hot = tree_hot
Category.get_depth_map = get_depth_map
Category.get_depth_map_array = get_depth_map_array