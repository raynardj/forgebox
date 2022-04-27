from PIL import Image
from typing import List, Callable
import math
from forgebox.html import data_url, DOM


def image_dom(
    img: Image,
    **kwargs
):
    """
    Create <img> tag with src='data:...'
        with PIL.Image object
    return forgebox.html.DOM
    """

    img_dom = DOM("", "img", dict(
        src=data_url(img),
    ))
    kwargs.update({"style": "padding:3px"})
    div_dom = DOM("", "div", kwargs)
    div_dom.append(img_dom)
    return div_dom


def with_title_dom(img: Image, title: str, col: int):
    from ..html import DOM
    img_dom = image_dom(img)
    out_dom = DOM(img_dom, "div", {"class": f"col-sm-{col}"})
    out_dom.append(title)
    return out_dom


def view_images(
    *images,
    num_per_row: int = 4,
    titles: List = None,
):
    """
    Create <div> wraping up images
    view_images(
        img1, img2, img3,
        img4, img5, img6,
        img7)()
    """
    from ..html import DOM
    frame = DOM("", "div", {"class": "row"})
    assert num_per_row in [1, 2, 3, 4, 6, 12],\
        "num_per_row should be in [1, 2, 3, 4, 6, 12]"
    col = 12//num_per_row

    if titles is not None:
        assert len(titles) == len(images),\
            f"title length:{len(titles)}!=image length:({len(images)})"
    if titles is None:
        for img in images:
            frame.append(image_dom(img, **{"class": f"col-sm-{col}"}))
    else:
        for img, title in zip(images, titles):
            frame.append(with_title_dom(img, title, col))
    return frame


class Subplots:
    """
    Simplifying plt sublots
    sub = Subplots(18)

    ```
    @sub.single
    def plotting(ax, data):
        ax.plot(data)

    for data in data_list:
        sub(data)
    ```
    """

    def __init__(self, total: int, ncol: int = 3, figsize=None):
        """
        total:int, total plot numbers
        ncol:int, number of columns
        figsize: Tuple, default (15, vertical_items*4)
        """
        from matplotlib import pyplot as plt

        self.i = 0
        self.size = list([math.ceil(float(total)/float(ncol)), ncol])
        self.ncol = ncol
        self.total = total
        self.figsize = figsize if figsize is not None else (15, self.size[0]*4)
        self.fig, self.ax = plt.subplots(*self.size, figsize=self.figsize)

    def __len__(self) -> int: return self.total

    def single(self, f: Callable) -> Callable:
        """
        Decorating a plt subplot function
        """
        self.single_func = f
        return self.single_func

    def __call__(self, *args, **kwargs):
        if self.i >= self.total:
            raise StopIteration(f"You said total would be {self.total}!")
        ax = self.ax[math.floor(
            float(self.i)/float(self.ncol)), self.i % self.ncol]
        self.single_func(ax, *args, **kwargs)
        self.i += 1
