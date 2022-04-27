__all__ = ['list_vc', 'col_list_vc', 'default_rename_rule', 'rename_by_rule', 'column_order']

import pandas as pd
from typing import Callable
import numpy as np


def list_vc(
    df, colname: str, value: str
) -> pd.DataFrame:
    """
    count the values in a column
        that each cell is a list
    """
    return df[colname].list_vc(value)

def col_list_vc(
    col, value: str
) -> pd.DataFrame:
    """
    count the values in a column
        that each cell is a list
    """
    return pd.DataFrame(
        col.apply(lambda x: value in x).value_counts()
    )

pd.DataFrame.vc = lambda self,col:pd.DataFrame(self[col].value_counts())
pd.Series.list_vc = col_list_vc
pd.DataFrame.list_vc = list_vc

def split(df, valid=0.2, ensure_factor=2):
    """
    df: dataframe
    valid: valid ratio, default 0.1
    ensure_factor, ensuring the row number to be the multiplication of this factor, default 2
    return train_df, valid_df
    """
    split_ = (np.random.rand(len(df)) > valid)
    train_df = df[split_].sample(frac=1.).reset_index().drop("index", axis=1)
    valid_df = df[~split_].sample(frac=1.).reset_index().drop("index", axis=1)

    if ensure_factor:
        train_mod = len(train_df) % ensure_factor
        valid_mod = len(valid_df) % ensure_factor
        if train_mod: train_df = train_df[:-train_mod]
        if valid_mod: valid_df = valid_df[:-valid_mod]
    return train_df, valid_df

pd.DataFrame.split = split


def default_rename_rule(x: str) -> str:
    return x.replace(" ", "_").replace("-", "_").lower()


def rename_by_rule(
    df,
    rule: Callable = default_rename_rule
) -> pd.DataFrame:
    """
    rename the columns by a rule function
    """
    df = df.rename(
        columns=dict((c, rule(c)) for c in df.columns))
    return df

pd.DataFrame.rename_by_rule = rename_by_rule


def column_order(df, *col_names) -> pd.DataFrame:
    """
    df = df.column_order("col1", "col2", "col3")
    will put col1, col2, and col3 as the 1st 3 column
    """
    cols = list(df.columns)

    for col_name in list(col_names)[::-1]:

        # warn if the column exist
        if col_name not in cols:
            print(f"Column:'{col_name}' not in dataframe")
            continue
        cols.insert(0, cols.pop(cols.index(col_name)))
    return df[cols]

pd.DataFrame.column_order = column_order

