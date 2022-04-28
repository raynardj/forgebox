# ForgeBox

[![PyPI version](https://img.shields.io/pypi/v/forgebox)](https://pypi.org/project/forgebox/)
![Python version](https://img.shields.io/pypi/pyversions/forgebox)
![License](https://img.shields.io/github/license/raynardj/forgebox)
![PyPI Downloads](https://img.shields.io/pypi/dm/forgebox)
[![pypi build](https://github.com/raynardj/forgebox/actions/workflows/publish.yml/badge.svg)](https://github.com/raynardj/forgebox/actions/workflows/publish.yml)
[![Test](https://github.com/raynardj/forgebox/actions/workflows/test.yml/badge.svg)](https://github.com/raynardj/forgebox/actions/workflows/test.yml)

> Data science comprehensive toolbox

## Installation

Easy simple installation in 1 line
```shell
pip install forgebox
```

If not specified, you need anaconda3 for most of the tools, python shold be at least >=3.6

## Features 🚀 Briefing

> This is a tool box with comprehensive **utilies**, to put it simply, I just hope most of my frequetyly used DIY tools in in place and can be easily installed and imported

### Lazy, fast imports 🤯

The following command will import many frequent tools for data science, like **pd** for pandas, **np** for numpy, os, json, PIL.Image for image processing

```python
from frogebox.imports import *
```

No more following verbosity
```python
import pandas as pd
import numpy as np
import os
import json
...
```
### Get a dataframe of file details under a  directory

```python
from forgebox.files import file_detail
```

```python
file_detail("/Users/xiaochen.zhang/.cache/").sample(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
      <th>file_type</th>
      <th>parent</th>
      <th>depth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>36</td>
      <td>/Users/xiaochen.zhang/.cache/torch/transformer...</td>
      <td>json</td>
      <td>transformers</td>
      <td>7</td>
    </tr>
    <tr>
      <td>13</td>
      <td>/Users/xiaochen.zhang/.cache/torch/transformer...</td>
      <td>json</td>
      <td>transformers</td>
      <td>7</td>
    </tr>
    <tr>
      <td>51</td>
      <td>/Users/xiaochen.zhang/.cache/langhuan/task_NER...</td>
      <td>json</td>
      <td>task_NER_210121_140513</td>
      <td>7</td>
    </tr>
    <tr>
      <td>32</td>
      <td>/Users/xiaochen.zhang/.cache/torch/transformer...</td>
      <td>lock</td>
      <td>transformers</td>
      <td>7</td>
    </tr>
    <tr>
      <td>58</td>
      <td>/Users/xiaochen.zhang/.cache/langhuan/task_Cla...</td>
      <td>json</td>
      <td>task_Classify_210128_164710</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>


### HTML in notebook

```python
from forgebox.html import DOM, list_group, list_group_kv
```

This will map a clear HTML table view of wild tree type json structure/ list

```python
bands = ["police", "headpin", {"ac":"dc"}]
list_group(bands)()
```

#### Coding html in python

```python
title = DOM("Title example","h5", kwargs={"style":"color:#3399EE"})
ul = DOM("","ul");
for i in range(5):
    ul = ul.append(DOM(f"Line {i}", "li", kwargs={"style":"color:#EE33DD"}))

title()
ul()
```


<h5 style="color:#3399EE">Title example</h5>



<ul><li style="color:#EE33DD">Line 0</li><li style="color:#EE33DD">Line 1</li><li style="color:#EE33DD">Line 2</li><li style="color:#EE33DD">Line 3</li><li style="color:#EE33DD">Line 4</li></ul>


### Interactive Widgets
> Interactive widgets work with in jupyter notebooks

#### Search box 🔎 for dataframe
This will create an interactive text input box to search through the pandas dataframe, within the columns you set.

if ```manual``` is set to False, the search will respond to **each of your key press**, it's fast but will suffer terrible user experience if the dataframe is huge in size.

```python
from forgebox.widgets import search_box

search_box(data_df, columns=["col1","col2"], manual=False)
```

#### paginate
You can browse through a pandas dataframe like fliping pages 📄.

```python
# this will import many things like enhanced pandas
from forgebox.imports import *
df  = pd.read_csv("xxxx.csv")
df.paginate()
```

```python
from forgebox.widgets import paginate

paginate(your_dataframe, page_len=10)
```
