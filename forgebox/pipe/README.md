# FrogeBox Pipe
### A pandas dataframe pipeline

This is a tutorial about tabular data processing pipeline


```python
import pandas as pd
import numpy as np
from pathlib import Path
import os
```

Sample dataset: [New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)


```python
HOME = Path(os.environ["HOME"])/"data"
DATA = HOME/"AB_NYC_2019.csv"
```

A preview on data set


```python
df = pd.read_csv(DATA)

df.head()
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
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>2787</td>
      <td>John</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>Private room</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>Skylit Midtown Castle</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3647</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>4632</td>
      <td>Elisabeth</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>Private room</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3831</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>1</td>
      <td>270</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>7192</td>
      <td>Laura</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### DataFrame Pipeline 


```python
from forgebox.pipe import Node,colEdge
from forgebox.pipe.edge import capMinMaxEdge,eng_twt_tk,fillNaEdge
```

Start node, a pandas dataframe


```python
start_node = Node(df)
```

#### Column edges
```edge``` defines a method of processing data.
* colEdge and its sub-class process column
* frameEdge and its cub_class process a dataframe (multiple columns)


```python
cap_minmax_edge = capMinMaxEdge(max_ = 100)
fill_na_edge = fillNaEdge("")
```

### Nodes/Edges relationship mapping
* node1 ==edge1==> node2 : 
    * 
    ```python
    node2 = node1 | edge1
    ```
* node1 ==edge1==>edge2==> node2 : 
    * 
    ```python
    node2 = node1 | edge1 | edge2
    ```
    * 
    ```python
    node3 = node2 | edge3
    ```
    
### Relationship mapping with **columns specified**
* Specifying 1 column
    * 
    ```python
    node2 = node1 | edge1%"column_1"
    ```
* Specifying multiple columns
    * 
    ```python
    node2 = node1 | edge1*["column_2","column_3","column_4"]
    ```

Setting the node/edge pipeline


```python
# clip minmax value on 1 column
# fill na to "", on 2 columns
# tokenize, on 2 columns
end_node = start_node|cap_minmax_edge %"number_of_reviews"\
                    |fill_na_edge *["name","room_type"]\
                    |eng_twt_tk*["name","room_type"]   
```

Print out the pipeline layout


```python
end_node
```




    <forge pipeline node>
    	|cap min:None max:100
    	|fillna_
    	|En Tokenization



Excute the processing


```python
end_df = end_node.run()
```

    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization


Checking the processed data


```python
end_df.head()
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
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539</td>
      <td>[Clean, &amp;, quiet, apt, home, by, the, park]</td>
      <td>2787</td>
      <td>John</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>[Private, room]</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>[Skylit, Midtown, Castle]</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>[Entire, home, /, apt]</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3647</td>
      <td>[THE, VILLAGE, OF, HARLEM, ..., NEW, YORK, !]</td>
      <td>4632</td>
      <td>Elisabeth</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>[Private, room]</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3831</td>
      <td>[Cozy, Entire, Floor, of, Brownstone]</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>[Entire, home, /, apt]</td>
      <td>89</td>
      <td>1</td>
      <td>100</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>[Entire, Apt, :, Spacious, Studio, /, Loft, by...</td>
      <td>7192</td>
      <td>Laura</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>[Entire, home, /, apt]</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Working with bigger data
There is often the cases when we have to deal with csv > 500m in size or a huge sql table.

Gladely, pandas offer a chunksize solution, so we can process huge structured data in batch(defined by the chunk size)


```python
from forgebox.pipe import chunkNode
from forgebox.pipe.edge import saveCSV, saveSQL
```

```saveCSV``` : saving file could be a part of the pipeline, we can follow that edge by ```saveSQL``` if we like


```python
start_node = chunkNode(pd.read_csv(DATA, chunksize=5000), verbose = 1)

end_node = start_node|cap_minmax_edge %"number_of_reviews"\
                    |fill_na_edge *["name","room_type"]\
                    |eng_twt_tk*["name","room_type"]\
                    |saveCSV(HOME/"nyc_processed.csv")
```

Pipeline layout summary


```python
end_node
```




    <forge pipeline node>
    	|cap min:None max:100
    	|fillna_
    	|En Tokenization
    	|save to csv



**Notice**
* ```run``` function that 
    * start with a chunkNode has **no return**, 
    * start with a Node will ruturn the **result dataframe**. 
* This feature is purposefully designed, assuming the result data could also be huge and not suitable to remain its entirety in RAM.

Excute the processing， if you are annoyed by the verbosity, set ```verbose=0```


```python
end_node.run()
```

    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv
    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:save to csv



```python
!head -n3 {HOME/"nyc_processed.csv"}
```

    0	2539	['Clean', '&', 'quiet', 'apt', 'home', 'by', 'the', 'park']	2787	John	Brooklyn	Kensington	40.647490000000005	-73.97237	['Private', 'room']	149	1	9	2018-10-19	0.21	6	365
    1	2595	['Skylit', 'Midtown', 'Castle']	2845	Jennifer	Manhattan	Midtown	40.75362	-73.98376999999999	['Entire', 'home', '/', 'apt']	225	1	45	2019-05-21	0.38	2	355
    2	3647	['THE', 'VILLAGE', 'OF', 'HARLEM', '...', 'NEW', 'YORK', '!']	4632	Elisabeth	Manhattan	Harlem	40.809020000000004	-73.9419	['Private', 'room']	150	3	0			1	365


### Define a new edge
Create a new edge

Define the processing function with the ```define``` decorator.

* col is a pandas data series, the concept of ```column``` in pandas
* In this case we use the ```apply``` function of data series, any decorated function would work as long as it return another data series


```python
lower_case = colEdge("lower case")


def lowerList(x):
    return list(str(i).lower() for i in x)

@lower_case.define
def lower(col):
    return col.apply(lowerList)
```


```python
# The DIYed edge will work on columns "name" and "room_type" after tokenization
df = pd.read_csv(DATA)
start_node = Node(df)
end_node = start_node|cap_minmax_edge %"number_of_reviews"\
                    |fill_na_edge *["name","room_type"]\
                    |eng_twt_tk*["name","room_type"]\
                    |lower_case*["name","room_type"]
```


```python
end_node
```




    <forge pipeline node>
    	|cap min:None max:100
    	|fillna_
    	|En Tokenization
    	|lower case




```python
end_df = end_node.run()
end_df.head()
```

    [df edge]:cap min:None max:100
    [df edge]:fillna_
    [df edge]:En Tokenization
    [df edge]:lower case





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
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539</td>
      <td>[clean, &amp;, quiet, apt, home, by, the, park]</td>
      <td>2787</td>
      <td>John</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>[private, room]</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>[skylit, midtown, castle]</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>[entire, home, /, apt]</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3647</td>
      <td>[the, village, of, harlem, ..., new, york, !]</td>
      <td>4632</td>
      <td>Elisabeth</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>[private, room]</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3831</td>
      <td>[cozy, entire, floor, of, brownstone]</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>[entire, home, /, apt]</td>
      <td>89</td>
      <td>1</td>
      <td>100</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>[entire, apt, :, spacious, studio, /, loft, by...</td>
      <td>7192</td>
      <td>Laura</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>[entire, home, /, apt]</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


