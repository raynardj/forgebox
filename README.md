# ForgeBox
> Data science comprehensive toolbox 🛠⚔️📦


![forgebox logo](https://raw.githubusercontent.com/raynardj/forgebox/new_feature/docs/logo.jpg)

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

No more🚫 following typings
```python
import pandas as pd
import numpy as np
import os
import json
...
```

### Categorical converter

> Mapping and converting categorical infomation

```python
from forgebox.category import Category
```

```python
az = list(map(chr,range(ord("A"), ord("z")+1)))
print(az)
```

    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


```python
cate_az = Category(az)
cate_az
```




    Category Manager with 58



```python
cate_az.c2i["R"], cate_az.i2c[17]
```




    (17, 'R')



```python
cate_az.c2i[list("ForgeBox")]
```




    array([ 5, 46, 49, 38, 36,  1, 46, 55])



```python
cate_az.i2c[[ 5, 46, 49, 38, 36,  1, 46, 55]]
```




    array(['F', 'o', 'r', 'g', 'e', 'B', 'o', 'x'], dtype='<U1')



Padding missing token

```python
cate_az = Category(az, pad_mst=True)
cate_az.c2i[list("Forge⚡️Box")]
```




    array([ 6, 47, 50, 39, 37,  0,  0,  2, 47, 56])



### Get a dataframe of file details under a  directory

```python
from forgebox.files import file_detail
```

```python
file_detail("/Users/xiaochen.zhang/.cache/").sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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



### JS style async

```python
from forgebox.asyncing import Async
from time import sleep
```

```python
def something_time_costing_but_you_dont_want_to_wait(x):
    sleep(x)
    return f"slept for {x} seconds"

def task2_you_will_perfrom_after_the_time_costing_one(x):
    print(f"[result]:\t{x}")
    return 1

print("1111111111")

Async(something_time_costing_but_you_dont_want_to_wait)(2)\
.then(task2_you_will_perfrom_after_the_time_costing_one)\
.catch(print)

print("22222222222")
```

    1111111111
    22222222222
    [result]:	slept for 2 seconds


### HTML in notebook

```python
from forgebox.html import DOM, list_group, list_group_kv
```

This will map a clear HTML table view of wild tree type json structure/ list

```python
bands = ["police", "headpin", {"ac":"dc"}]
list_group(bands)()
```


<ul class="list-group"><li class="list-group-item">police</li><li class="list-group-item">headpin</li><li class="list-group-item"><ul class="list-group"><li class="list-group-item"><div class="row"><strong class="col-sm-5">ac</strong><span class="col-sm-7">dc</span></div></li></ul></li></ul>


```python
questions = {
    "question":"answer",
    "another":{
        "deeper question": "answer again"},
    "final":{
        "questions": ["what","is","the","answer", "to",
            ["life", "universe","everything"]]}
}
list_group_kv(questions)()
```


<ul class="list-group"><li class="list-group-item"><div class="row"><strong class="col-sm-5">question</strong><span class="col-sm-7">answer</span></div></li><li class="list-group-item"><div class="row"><strong class="col-sm-5">another</strong><span class="col-sm-7"><ul class="list-group"><li class="list-group-item"><div class="row"><strong class="col-sm-5">deeper question</strong><span class="col-sm-7">answer again</span></div></li></ul></span></div></li><li class="list-group-item"><div class="row"><strong class="col-sm-5">final</strong><span class="col-sm-7"><ul class="list-group"><li class="list-group-item"><div class="row"><strong class="col-sm-5">questions</strong><span class="col-sm-7"><ul class="list-group"><li class="list-group-item">what</li><li class="list-group-item">is</li><li class="list-group-item">the</li><li class="list-group-item">answer</li><li class="list-group-item">to</li><li class="list-group-item"><ul class="list-group"><li class="list-group-item">life</li><li class="list-group-item">universe</li><li class="list-group-item">everything</li></ul></li></ul></span></div></li></ul></span></div></li></ul>


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


### Free style mapping

Works on every value of a complicated dictionary structure (eg. list in dict in list in dict, etc,. 😳)

```python
from forgebox.freemap import FreeMap

# flatten decides if we want to flatten the strucuture
freemap_tool = FreeMap(
    <function/callable applying to every value>,
    <function/callable that filters every value>,
    flatten=True
)

data2 = freemap_tool(data1)
```

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
from forgebox.widgets import paginate

paginate(your_dataframe, page_len=10)
```

#### Single button callback
> a fully functional page with a single button, this single button is bonded to a function

This is as much code as you need, to build a fully functional interactive page shows sql table from jupyter, that you can:* choose which table to visit* choose how many lines you want to show, (with a slider)
* configure the where condition with a text box on front end

```python
tablename_list = ["pubmed", "patient", "users", "drugs"]

from forgebox.html import DOM
def show_sql_table(sql_input:str) -> str:
    with engine.connect() as conn:
        df=pd.read_sql(sql_input, con=conn)
    # display the table as html
    DOM(df.to_html(),"div")()

@SingleButton(callback=show_sql_table)
def abc(
    limit:{"typing":int, "default":10, "min":5, "max":20},
    where_condition:{"typing":str, "default": "where 1=1", },
    table:{"typing":list, "options":tablename_list}
):
    return f"SELECT * FROM {table} {where_condition} LIMIT {limit}"
```
