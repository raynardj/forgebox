# ForgeBox
> Data science comprehensive toolbox 🛠⚔️📦


![logo](nbs/logo.jpg)

## Installation

Easy simple installation in 1 shot
```shell
pip install forgebox
```

If not specified, you need anaconda3 for most of the tools, python shold be at least >=3.6

## Features 🚀 Briefing

> This is a tool box with comprehensive utilies, to put it simply, I just hope most of my frequetyly used DIY tools in in place and can be easily installed and imported

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

This is as much code as you need, to build a fully functional interactive page shows sql table from jupyter, that you can:✅ choose which table to visit✅ choose how many lines you want to show, (with a slider)✅ configure the where condition with a text box on front end
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
    return f"sql > SELECT * FROM {table} {where_condition} LIMIT {limit}"
```

### Dataset Layering 🍰
> Instead of creating a bunch of dataset just treat another dataset as extra layer of __getitem__ function

```python
@layering(SomeStringDataset, "tensorDataset")
def tokenizing(x):
    return tokenizer(x, return_tensors='pt')['input_ids'][0]

@layering(tokenizing, "guessNextDataset")
def guess_next(x):
    return x[:-1], x[1:]

some_string_data = SomeStringDataset()

tokenized_data = some_string_data.next_layer()
print(tokenized_data[3])

guess_pair = tokenized_data.next_layer()
print(guess_pair[3])
```

And you can test these layer functions one by one