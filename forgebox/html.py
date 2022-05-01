import math
from PIL.Image import Image as ImageClass
import base64
from io import BytesIO
from IPython.display import HTML
from typing import Dict, Any

try:
    display
except:
    display = print


class DOM:
    """
    DOM object in python
    dom = DOM("some text", "div", {"class":"myclass"})
    dom.append(another_dom)
    dom['class'] = "newclass"
    # show in jupyter
    dom()
    """

    def __init__(self, txt, tag, kwargs=dict()):
        self.txt = txt
        self.tag = str(tag).lower()
        self.attrs = kwargs
        self.refresh_attr()

    @staticmethod
    def extend(text, tag, **kwargs):
        attributes = (" ".join(f'{k}="{v}"' for k, v in kwargs.items()))
        attributes = " "+attributes if attributes else attributes
        start = f"<{tag}{attributes}>"
        inner = f"{text}"
        end = f"</{tag}>"
        text = f"{start}{inner}{end}"
        return start, inner, end

    def refresh_attr(self):
        self.start, self.inner, self.end = self.extend(
            self.txt, self.tag, **self.attrs)

    def __mul__(self, new_tag):
        assert type(new_tag) == str
        return DOM(self.text, new_tag)

    def __add__(self, dom):
        return self.text+dom.text

    def __repr__(self):
        return f"{self.start}{self.inner}{self.end}"

    def __getitem__(self, k):
        return self.attrs[k]

    def __setitem__(self, k, v):
        self.update({k, v})

    def __call__(self):
        self.display()

    @property
    def text(self):
        return str(self)

    def append(self, subdom):
        self.inner = self.inner+str(subdom)
        return self

    def update(self, dict_):
        self.attrs.update(dict_)
        self.refresh_attr()
        return self

    def display(self):
        """
        Display the html in Jupyter
        """
        display(HTML(self.text))


def image_to_base64(
    img: ImageClass
) -> str:
    """
    Transform PIL Image to base64 for API
    Return:
        - base64 encoded image bytes
    """
    img = img.convert('RGB')
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()


def data_url(
    img: ImageClass
) -> str:
    """
    Return:
        - data url string,
            can be used as the src value of <img>
    """
    return f"data:image/jpg;base64,{image_to_base64(img)}"


def img_dom(img: ImageClass) -> DOM:
    """
    show PIL.Image in HTML
    """
    return DOM("", "img", {"src": data_url(img)})


def deeper(x):
    if type(x) in [list, set, tuple]:
        return list_group(x)
    if type(x) == dict:
        return list_group_kv(x)
    if type(x) in [int, str, float, bool]:
        return x
    if type(x) == ImageClass:
        return img_dom(x)
    return x


def list_group(iterable):
    ul = DOM("", "ul", {"class": "list-group"})
    for i in iterable:
        li = DOM(deeper(i), "li", {"class": "list-group-item"})
        ul.append(li)
    return ul


def col_sm(iterable, portions=None,):
    if portions == None:
        portions = [math.floor(12/len(iterable)), ] * len(iterable)
    row = DOM("", "div", {"class": "row"})
    for i, p in zip(iterable, portions):
        row.append(DOM(i, "div", {"class": f"col-sm-{p}"}))
    return row


def list_group_kv(data: Dict[str, Any]) -> DOM:
    """
    Key
    """
    result = []
    for k, v in data.items():
        row = DOM("", "div", {"class": "row"})
        row.append(DOM(f"{k}", "strong", {"class": "col-sm-5"}))\
            .append(DOM(f"{deeper(v)}", "span", {"class": "col-sm-7"}))
        result.append(row)
    return list_group(result)


def JS(code):
    DOM(code, "script",)()


def JS_file(path):
    """
    load javascript file
    """
    with open(path, "r") as f:
        DOM(f.read(), "script")()
