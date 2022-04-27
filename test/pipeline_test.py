from forgebox.cosine import CosineSearch
from forgebox.config import Config
from forgebox.files import file_detail
from forgebox.batcher import Batcher
from forgebox.imports import *


def test_batecher():
    batcher = Batcher([1, 2, 3, 4, 5], batch_size=2)
    batches = list(batcher)
    assert len(batches) == 3
    assert batches[0] == [1, 2]
    assert batches[1] == [3, 4]
    assert batches[2] == [5]
    assert batcher.one_batch() == [1, 2]


def test_vc():
    df = pd.DataFrame(dict(
        col1=["a", "b", "c", "a", "b", "c"],
        col2=["a", "b", "c", "a", "b", "c"],
        col3=["a", "b", "c", "a", "b", "c"]
    ))
    vc = df.vc("col1")
    assert vc.iloc[0, 0] == 2


def test_split():
    df = pd.DataFrame(dict(
        col1=list(range(50))))
    train_df, valid_df = df.split()
    assert len(train_df) > len(valid_df)
    train_df, valid_df = df.split(.8)
    assert len(train_df) < len(valid_df)


def test_column_order():
    df = pd.DataFrame(dict(
        col1=["a", "b", "c", "a", "b", "c"],
        col2=["a", "b", "c", "a", "b", "c"],
        col3=["a", "b", "c", "a", "b", "c"]
    ))
    df = df.column_order("col2", "col1", "col3")
    assert df.columns.tolist() == ["col2", "col1", "col3"]


def test_get_files():
    import os
    p = Path(os.getcwd())
    test_dir = p/"test_dir"
    test_dir.mkdir(exist_ok=True)
    test_dir.joinpath("test_file.txt").touch()
    test_dir.joinpath("test_file2.txt").touch()
    test_dir.joinpath("test_file3.txt").touch()
    test_dir.joinpath("test_file4.txt").touch()
    df = file_detail(test_dir)
    assert len(df) == 4
    assert df.file_type.iloc[0] == "txt"
    assert df.file_type.iloc[1] == "txt"
    assert df.file_type.iloc[2] == "txt"
    assert df.file_type.iloc[3] == "txt"
    assert df.parent.iloc[0] == "test_dir"

    # clear test_dir

    for file in df.path:
        os.remove(file)
    test_dir.rmdir()


def test_config():
    config = Config(a=1, b=2, c=[1, "2"])
    config.save("test_config.json")
    config2 = Config.load("test_config.json")
    assert config2.a == 1
    assert config2.b == 2
    assert config2.c == [1, "2"]


def test_cosine_search():
    base = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    vec = np.array([4, 5, 2])
    vec2 = np.array([2, 9, 9])
    cos = CosineSearch(base)
    assert (cos(vec) == [2, 1, 0]).all()
    assert (cos(vec2) == [0, 1, 2]).all()
    assert (cos.search(vec) == [2, 1 ,0]).all()
