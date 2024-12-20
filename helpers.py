from __future__ import annotations

from datetime import datetime
from errno import EEXIST
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from corpora import Matrix


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        print(f"func: {f.__name__} took: {te-ts}")
        return result

    return wrap


def style(v, minus=True, props=""):
    if isinstance(v, str):
        return
    if minus:
        return props if v < 0 else None
    else:
        return props if v >= 0 else None


def plus_minus(df):
    return df.style.applymap(
        style, minus=True, props="background-color:#e8aea9;color:black;"
    ).applymap(style, minus=False, props="background-color:#a9b4eb;color:black;")


def write(
    path: Path, table: pd.DataFrame | np.array, name: str, color: None | bool = None
):
    if isinstance(table, pd.DataFrame):
        if color:
            table.pipe(plus_minus).to_excel(path / f"{name}.xlsx", index=False)
        else:
            table.to_csv(path / f"{name}.csv", index=False)
    else:
        with open(path / f"{name}.npy", "wb") as f:
            np.save(f, table)
    print(f"wrote {name}")


def read(
    path: Path, name_with_ext: str, parse_date: str = "date"
) -> pd.DataFrame | np.array:
    if name_with_ext[-3:] == "npy":
        return np.load(path / name_with_ext)
    else:
        return pd.read_csv(
            path / name_with_ext,
            parse_dates=([parse_date] if name_with_ext != "dvr.csv" else []),
        )
