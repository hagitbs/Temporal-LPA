from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import json

import numpy as np
import pandas as pd
from scipy.special import lambertw
import bottleneck as bn

from algo import entropy
from helpers import read, write, timing


class Matrix:
    def __init__(self, matrix: np.array):
        self.matrix = matrix

    def _get_epsilon(self, lambda_=1):
        """
        λ is the contibution to the entropy by the terms with probability ε
        ε ≈ six orders of magnitude smaller than λ
        """
        m = np.count_nonzero((self.matrix == 0), axis=1).max()
        if lambda_ > m / (np.e * np.log(2)) or lambda_ <= 0:
            raise ValueError
        s = entropy(self.matrix).sum(axis=1).max()
        res = np.minimum(
            np.e ** lambertw(-lambda_ * np.log(2) / m, k=-1).real, 2 ** (-s)
        )
        return res

    def epsilon_modification(
        self,
        epsilon: float | None = None,
        lambda_: float | int = 1,
        threshold: float = 0,
    ) -> Matrix:
        if not epsilon:
            epsilon = self._get_epsilon(lambda_)
        beta = 1 - epsilon * np.count_nonzero(self.matrix <= threshold, axis=1)
        matrix = self.matrix * beta[:, None]
        matrix[matrix <= threshold] = epsilon
        return Matrix(matrix)

    def apply(
        self, metric: str, save: bool = False, path: None | Path = None
    ) -> pd.DataFrame:
        res = []
        func = getattr(import_module("algo"), metric)
        # TODO: apply_along_axis or something
        for i in range(len(self.matrix) - 1):
            res.append(func(self.matrix[i : i + 2]))
        res_df = (
            pd.DataFrame({metric: res}).reset_index().rename(columns={"index": "date"})
        )
        if save:
            write(path, (res_df, metric))
        return res_df

    def delete(self, ix, axis):
        self.matrix = np.delete(self.matrix, obj=ix, axis=axis)

    def normalized_average_weight(self) -> np.array:
        average_weight = bn.nanmean(self.matrix, axis=0)
        return average_weight / average_weight.sum()

    def moving_average(self, window: int) -> np.array:
        max_ = bn.nanmax(self.matrix, axis=1)
        min_ = bn.nanmin(self.matrix, axis=1)
        ma = bn.move_mean(bn.nanmean(self.matrix, axis=1), window=window, min_count=1)
        return pd.DataFrame({"ma": ma, "max": max_, "min": min_}).reset_index()


class Corpus:
    def __init__(self, date_cat: pd.Series | pd.DatetimeIndex, element_cat: pd.Series):
        self.date_cat = pd.Categorical(date_cat, ordered=True).dtype
        self.element_cat = pd.Categorical(element_cat, ordered=True).dtype

    def update_dates(self, date):
        self.date_cat = pd.CategoricalDtype(
            self.date_cat.categories[~self.date_cat.categories.isin([date])],
            ordered=True,
        )

    def code_to_cat(self, code: str, what="date") -> int:
        return getattr(self, f"{what}_cat").categories[code]

    def pivot(self, data: pd.DataFrame) -> Matrix:
        d = data["date"].astype(self.date_cat)
        e = data["element"].astype(self.element_cat)
        idx = np.array([d.cat.codes, e.cat.codes]).T
        matrix = np.zeros(
            (len(d.cat.categories), len(e.cat.categories)), dtype="float64"
        )
        matrix[idx[:, 0], idx[:, 1]] = data["global_weight"]
        return Matrix(matrix[min(d.cat.codes) : max(d.cat.codes) + 1])

    # def save(self, path: Path):
    #     with open(path / "corpus.json", "w") as fp:
    #         d = {
    #             "date": self.date_cat.categories.astype(str).to_list(),
    #             "elements": self.element_cat.categories.astype(str).to_list(),
    #         }
    #         json.dump(d, fp)

    # @staticmethod
    # def load(path: Path) -> Corpus:
    #     with open(path / "corpus.json") as f:
    #         data = json.load(f)
    #     return Corpus(pd.to_datetime(data["date"]), pd.Series(data["elements"]))
