from ipywidgets import (
    VBox, Button,HTML,
    FloatSlider, IntSlider, SelectionSlider, Dropdown, Label, Checkbox,
    interact, Output, interact_manual)

from forgebox.imports import *
from forgebox.html import DOM
from abc import ABC
from typing import List, Callable

try:
    display
except:
    display = print


def pct_to_float(x):
    if type(x) != str:
        return x
    if "%" not in x:
        return x
    else:
        return float(x[:-1])/100


def ensure_pct(df):
    for col in df:
        if df[col].dtype.name == "object":
            df[col] = df[col].apply(pct_to_float)
    return df


def detect_number_column(df):
    """
    Detect number columns in dataframe
    """
    cols = df.columns
    dtypes = [df[col].dtype.name for col in cols]
    return pd.DataFrame({"cols": cols, "dtypes": dtypes})


class DataFilter:
    """
    Single column number filter
    """

    def __init__(self, df: pd.DataFrame, fix_pct=True):
        """
        df: input dataframe

        data_filter = DataFilter(df)

        # start filtering
        data_filter()
        """
        self.df = df

        if fix_pct:
            self.df = ensure_pct(self.df)

    def show_distribution(self, col_name):
        """
        show distribution of a column, using plotly
        """
        import plotly.express as px
        fig = px.histogram(self.df, x=col_name, height=300, width=800)
        return fig

    def create_filter(self, field: str) -> None:
        big_boxes = []

        dtype = self.df[field].dtype.name
        if 'float' in dtype:
            slider = FloatSlider
            slide = slider(
                min=self.df[field].min(),
                max=self.df[field].max(),
                step=0.001
            )
        elif 'int' in dtype:
            slider = IntSlider
            slide = slider(
                min=self.df[field].min(),
                max=self.df[field].max(),)
        else:
            print(f"filter of {dtype} not supported")
            slider = SelectionSlider
            slide = slider(options=sorted(map(str, set(list(self.df[field])))))

        btn = Button(description="Run Filter")
        btn.on_click(self.execute_filter)

        print(f"NaN count: {(self.df[field].isna()).sum()}")

        widget = VBox([
            Label(f"Range for {field}"),
            Dropdown(options=["Larger Than or equal to",
                              "Smaller Than or equal to"]),
            slide,
            Checkbox(description="Remove NaN", value=True),
            btn
        ])
        self.widget = widget
        widget.original_name = field

        display(widget)

    def execute_filter(
            self, _) -> None:
        """
        This function will be used as a callback
        for ipywidgets.Button.on_click
        """
        original_name = self.widget.original_name
        label_, condi_, value_, remove_na_, btn_ = self.widget.children
        label, condi, value, remove_na = label_.value, condi_.value, value_.value, remove_na_.value
        condi = ">=" if condi == 'Larger Than or equal to' else "<="
        if type(value_) == SelectionSlider:
            value = f"'{value}'"
        expression = f"{original_name} {condi} {value}"

        if remove_na:
            self.remove_na(original_name)

        print(f"Filter with query expression: {expression}")
        before = len(self.df)
        self.df = self.df.query(expression).reset_index(drop=True)
        after = len(self.df)
        print(f"[Before]: {before}, [After]: {after}")

    def remove_na(self, field):
        """
        Remove nan value in a dataframe
        """
        before = len(self.df)
        self.df = self.df[~self.df[field].isna()]
        after = len(self.df)
        print(f"Remove NA on {field} [Before]: {before}, [After]: {after}")

    def __call__(self, columns=None):
        """
        Execute an interact to filter things column by column
        """
        columns = columns if columns else list(self.df.columns)

        @interact
        def select_field(field=columns):
            # visualize histogram
            self.show_distribution(field).show()

            # create a filter execution interactive
            self.create_filter(field)

# export


class LayerTorch:
    """
    information passon from layer to layer
    """

    def __init__(self, df, level=0, last_layer=None):
        self.df = df
        self.level = level
        self.data = dict()
        if last_layer is not None:
            self.last_layer = last_layer
        self.out = Output()
        if last_layer is None:
            self.axis = dict()
            df.filter_layers = self.axis
        else:
            self.axis = last_layer.axis
        self.axis[level] = self

    def __call__(self, **kwargs):
        self.data.update(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"Level:{self.level}\n\t{self.data}"

    def next_layer(self):
        new = LayerTorch(self.df, level=self.level+1, last_layer=self)
        self.next_layer = new
        return new

    @property
    def layers(self) -> pd.DataFrame:
        """
        Filter layers as a pandans dataframe
        """
        return pd.DataFrame(
            list(i.data for i in self.axis.values()))

    @property
    def filter_chain(self) -> str:
        query_list = list(i.data["query"] for i in self.axis.values())
        return ' and '.join(list(f"({q})" for q in query_list))


class RecursiveFilterCore(ABC):

    def display_queries(self, this_layer):
        if this_layer.level > 0:
            DOM("Queried Filters", "h3", )()
            display(this_layer.layers[:-1])

    def handpick(
        self,
        chunk_callbacks: List[Callable] = [],
        show_top: bool = True,
        show_top_k: int = 20,
        pick_value_top_k: int = 30,
        from_last_layer: LayerTorch = None
    ) -> None:
        """
        Hand pick the portion of the data frame you liked
            from filtering the column by value.
        A function from enhanced pandas dataframe

        Inputs:
        - chunk_callbacks: List[Callable]=[],
        - show_top: bool, default True, do we should
            the most frequent values of the current column
        - show_top_k: int, the number of rows we show for
            the most frequent values, when show_top=True,
            default 20
        - pick_value_top_k: int, number of the most frequent
            values in pick drop down default 30
        - from_last_layer: LayerTorch, default None, this
            column doesn't mean for user configuration
        """
        this_layer = LayerTorch(
            self) if from_last_layer is None else from_last_layer.next_layer()
        display(this_layer.out)

        with this_layer.out:
            self.display_queries(this_layer)

            DOM("Select Filter Column", "h3")()

            @interact
            def select_columns(column=self.columns):

                this_layer(column=column)
                series = self[column]
                vc = self.vc(column)

                if show_top:
                    display(vc.head(show_top_k))

                    this_layer.out.played = True

                top_values = list(vc.index)
                if pick_value_top_k is not None:
                    top_values = top_values[:pick_value_top_k]

                DOM(f"'{column}' equals to ?", "h3", )()

                @interact()
                def pick_value(picked=top_values):
                    query = f"`{column}`=='{picked}'"
                    sub = RecursiveFilter(self.query(query))

                    # keep record on this layer
                    this_layer(
                        query=query,
                        picked=picked,
                        before_rows=len(self),
                        after_rows=len(sub)
                    )

                    for cb in chunk_callbacks:
                        cb(sub)

                    @interact_manual
                    def start_recursion():
                        this_layer.out.clear_output()

                        with this_layer.out:

                            # Recursion
                            # Go on the filter to the next layer
                            sub.handpick(
                                chunk_callbacks=chunk_callbacks,
                                show_top=show_top,
                                show_top_k=show_top_k,
                                pick_value_top_k=pick_value_top_k,
                                from_last_layer=this_layer,
                            )

                    sub.paginate(10)


class RecursiveFilter(pd.DataFrame, RecursiveFilterCore):
    """
    Interactive Pandas DataFrame Filter
    df = RecursiveFilter(df)
    df.handpick()

        Hand pick the portion of the data frame you liked
            from filtering the column by value.
        A function from enhanced pandas dataframe

        Inputs:
        - chunk_callbacks: List[Callable]=[],
        - show_top: bool, default True, do we should
            the most frequent values of the current column
        - show_top_k: int, the number of rows we show for
            the most frequent values, when show_top=True,
            default 20
        - pick_value_top_k: int, number of the most frequent
            values in pick drop down default 30
        - from_last_layer: LayerTorch, default None, this
            column doesn't mean for user configuration
    """
    pass
