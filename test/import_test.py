def test_import_things():
    from forgebox.imports import (
        pd, np, json
    )
    assert hasattr(pd.DataFrame, "split")
    assert hasattr(pd.DataFrame, "paginate")

def test_import_category():
    from forgebox.category import (
        Category, MultiCategory, FastCategory, FastMultiCategory
    )