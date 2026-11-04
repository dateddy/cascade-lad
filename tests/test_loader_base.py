import pandas as pd
import pytest

from cascadelad.data.loaders import base
from cascadelad.data.loaders.base import (
    STANDARD_COLUMNS,
    LogLoader,
    get_loader,
    register_loader,
)


@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(base._REGISTRY)
    yield
    base._REGISTRY.clear()
    base._REGISTRY.update(saved)


def _frame(**overrides) -> pd.DataFrame:
    data = {
        "timestamp": pd.to_datetime(["2008-11-09 20:35:18", "2008-11-09 20:35:19"]),
        "content": ["a", "b"],
        "group_id": ["blk_1", "blk_2"],
        "label": [0, 1],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _register_dummy(name: str, df: pd.DataFrame) -> None:
    @register_loader(name)
    class Dummy(LogLoader):
        def _load(self) -> pd.DataFrame:
            return df


def test_registered_loader_round_trips():
    df = _frame()
    _register_dummy("dummy", df)
    out = get_loader("dummy", {"k": 1}).load()
    assert tuple(out.columns) == STANDARD_COLUMNS
    pd.testing.assert_frame_equal(out, df)


def test_duplicate_name_raises():
    _register_dummy("dup", _frame())
    with pytest.raises(ValueError, match="already registered"):
        _register_dummy("dup", _frame())


def test_unknown_name_raises_keyerror_listing_known():
    _register_dummy("known", _frame())
    with pytest.raises(KeyError, match="known"):
        get_loader("nope", {})


def test_wrong_or_extra_columns_raise():
    _register_dummy("extra", _frame().assign(extra=1))
    _register_dummy("missing", _frame().drop(columns="content"))
    _register_dummy("order", _frame()[["content", "timestamp", "group_id", "label"]])
    for name in ("extra", "missing", "order"):
        with pytest.raises(ValueError, match="expected columns"):
            get_loader(name, {}).load()


def test_label_outside_zero_one_raises():
    _register_dummy("badlabel", _frame(label=[0, 2]))
    with pytest.raises(ValueError, match="label"):
        get_loader("badlabel", {}).load()
