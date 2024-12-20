from __future__ import annotations
from copy import copy

from importlib import import_module
from pathlib import Path
from typing import List, Literal, Tuple
from itertools import combinations_with_replacement as cwr
import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.spatial.distance import cdist
from sklearn import decomposition as skd
from sklearn.preprocessing import StandardScaler
import bottleneck as bn

from algo import entropy, symmetrized_KLD, JSD
from helpers import read, write, timing


class Matrix:
    def __init__(self, matrix: np.array):
        self.matrix = matrix
        self.normalized = False

    def __bool__(self):
        return True if hasattr(self, "matrix") else False

    def _get_epsilon(self, lambda_=1) -> float:
        """
        λ is the contibution to the entropy by the terms with probability ε
        ε ≈ six orders of magnitude smaller than λ
        """
        m = np.count_nonzero((self.matrix == 0), axis=1).max()
        if m == 0:
            return 0
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
    ):
        if not epsilon:
            epsilon = self._get_epsilon(lambda_)
            print(epsilon)
            if epsilon == 0:
                return
        beta = 1 - epsilon * np.count_nonzero(self.matrix <= threshold, axis=1)
        self.matrix = self.matrix * beta[:, None]
        self.matrix[self.matrix <= threshold] = epsilon

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

    def normalize(self):
        self.normalized = True
        self.matrix = (self.matrix.T / self.matrix.sum(axis=1)).T

    def create_dvr(self, mean=False):
        if self.normalized:
            raise ValueError("Cannot create the DVR from normalized frequency data")
        if mean:
            self.dvr = self.normalized_average_weight()
        else:
            self.dvr = self.normalized_weight()

    def normalized_average_weight(self) -> np.ndarray:
        x = (self.matrix.T / self.matrix.sum(axis=1)).T
        return bn.nanmean(x, axis=0)

    # def normalized_average_weight(self) -> np.ndarray:
    #     average_weight = bn.nanmean(self.matrix, axis=0)
    #     return average_weight / average_weight.sum()

    def normalized_weight(self) -> np.ndarray:
        return self.matrix.sum(axis=0) / self.matrix.sum()

    def moving_average(self, window: int) -> np.array:
        max_ = bn.nanmax(self.matrix, axis=1)
        min_ = bn.nanmin(self.matrix, axis=1)
        ma = bn.move_mean(bn.nanmean(self.matrix, axis=1), window=window, min_count=1)
        return pd.DataFrame({"ma": ma, "max": max_, "min": min_}).reset_index()


class Corpus:
    def __init__(
        self,
        freq: pd.DataFrame | None = None,
        document_cat: pd.Series | pd.DatetimeIndex | None = None,
        element_cat: pd.Series | None = None,
        name: str | None = None,
    ):
        if (
            isinstance(freq, type(None))
            and isinstance(document_cat, type(None))
            and isinstance(element_cat, type(None))
        ):
            raise ValueError(
                "Either use a frequency dataframe or two series, one of document ids and one of elements"
            )
        elif isinstance(freq, pd.DataFrame):
            self.freq = freq
            document_cat = freq["document"]
            element_cat = freq["element"]
        self.document_cat = pd.Categorical(document_cat, ordered=True).dtype
        self.element_cat = pd.Categorical(element_cat, ordered=True).dtype
        if name:
            self.name = name

    def current(self, m=True):
        if hasattr(self, "signature_matrix"):
            curr = self.prevalent_matrix
        elif hasattr(self, "prevalent_matrix"):
            curr = self.prevalent_matrix
        elif hasattr(self, "distance_matrix"):
            curr = self.distance_matrix
        return curr.matrix if m else curr

    def __len__(self):
        """Number of documents"""
        return len(self.matrix.matrix)

    def update_documents(self, document):
        self.document_cat = pd.CategoricalDtype(
            self.document_cat.categories[
                ~self.document_cat.categories.isin([document])
            ],
            ordered=True,
        )

    def code_to_cat(self, code: str, what="document") -> int:
        return getattr(self, f"{what}_cat").categories[code]

    def pivot(self, freq: pd.DataFrame | None = None) -> Matrix:
        if hasattr(self, "freq"):
            freq = self.freq
        d = freq["document"].astype(self.document_cat)
        e = freq["element"].astype(self.element_cat)
        idx = np.array([d.cat.codes, e.cat.codes]).T
        matrix = np.zeros(
            (len(d.cat.categories), len(e.cat.categories)), dtype="float64"
        )
        matrix[idx[:, 0], idx[:, 1]] = freq["frequency_in_document"]
        return Matrix(matrix[min(d.cat.codes) : max(d.cat.codes) + 1])

    def create_dvr(
        self, equally_weighted: bool = False, matrix: None | Matrix = None
    ) -> pd.DataFrame:
        if not matrix:
            self.matrix = self.pivot(self.freq)
            matrix = self.matrix
        matrix.create_dvr(mean=equally_weighted)
        dvr = (
            pd.DataFrame(
                {
                    "element": self.element_cat.categories,
                    "global_weight": matrix.dvr,
                }
            )
            .reset_index()
            .rename(columns={"index": "element_code"})
            .sort_values("global_weight", ascending=False)
            .reset_index(drop=True)
        )
        return dvr[["element", "global_weight"]]

    def _prevalent_matrix(self, temporary_array):
        self.prevalent_matrix = copy(self.distance_matrix)
        self.prevalent_matrix.matrix[
            temporary_array & (self.distance_matrix.matrix < 0)
        ] = 0

    def _signature_matrix(self, sig_length, distances_df):
        # annuls all values that shouldn't appear in the signatures
        self.signature_matrix = Matrix(self.current().copy())  # copy?
        if sig_length:
            argsort = np.argsort(np.abs(self.signature_matrix.matrix), axis=1)
            indices = argsort[:, -sig_length:]
            p = np.zeros_like(self.signature_matrix.matrix)
            for i in range(p.shape[0]):
                p[i, indices[i]] = self.signature_matrix.matrix[i, indices[i]]
            self.signature_matrix.matrix = p
        signatures = [
            sig[1][self.signature_matrix.matrix[i] != 0].sort_values(
                key=lambda x: abs(x), ascending=False
            )
            for i, sig in enumerate(distances_df.iterrows())
        ]
        return signatures

    def _most_significant(self, most_significant, distances_df):
        sort = np.argsort(
            np.abs(self.distance_matrix.matrix).sum(axis=0), kind="stable"
        )[-most_significant:][::-1]
        # max_distances = [dist for _, dist in max_distances_df.iterrows()]
        return distances_df.iloc[:, sort]

    def avg_distance(self):
        avg_vecs = (self.matrix.matrix + self.matrix.dvr) / 2
        symmetrized_KLD(avg_vecs, self.matrix.dvr) + symmetrized_KLD(
            self.matrix.matrix, avg_vecs
        )

    def create_signatures(
        self,
        epsilon: float | None = None,
        most_significant: int | None = 30,
        sig_length: int | None = 500,
        prevalent: int | None = None,
        distance: str = "JSD",
        alpha: float | None = 0.5,
    ) -> List[pd.DataFrame] | Tuple[List[pd.DataFrame]]:
        """
        most_significant: checks which elements had the largest distance altogether and returns a dataframe consisting only of those distances, sorted
        """
        if sig_length == 0:
            sig_length = None
        if not hasattr(self, "matrix"):
            raise AttributeError("Please create dvr before creating signatures.")
        if not self.matrix.normalized:
            self.matrix.normalize()
        if prevalent:
            temporary_array = np.count_nonzero(self.matrix.matrix, axis=0) <= prevalent
        if distance == "KLD":
            dm = symmetrized_KLD(self.matrix.matrix, self.matrix.dvr)
        elif distance == "JSD":
            dm = JSD(self.matrix.matrix, self.matrix.dvr, alpha)
        elif distance == "KLDe":
            self.matrix.epsilon_modification(epsilon)
            dm = symmetrized_KLD(self.matrix.matrix, self.matrix.dvr)
        self.distance_matrix = Matrix(dm)
        if prevalent:
            self._prevalent_matrix(temporary_array)
        distances_df = pd.DataFrame(
            self.current(),
            index=self.document_cat.categories,
            columns=self.element_cat.categories,
        )
        res = [self._signature_matrix(sig_length, distances_df)]
        if most_significant:
            res.append(self._most_significant(most_significant, distances_df))
        if prevalent:
            res.append(temporary_array)
        return tuple(res)


def sockpuppet_distance(
    corpus1: Corpus,
    corpus2: Corpus,
    res: Literal["table", "matrix"] = "table",
    heuristic: bool = False,
    group_by: str | None = None,
    reduce_df: pd.DataFrame | None = None,
    distance: str = "cityblock",
) -> pd.DataFrame:
    matrices = []
    for c in [corpus1, corpus2]:
        matrix = copy(c.signature_matrix)
        matrix.matrix = matrix.matrix[:, ~np.all(matrix.matrix == 0, axis=0)]
        if heuristic:
            matrix.matrix[matrix.matrix > 0] += 1
            matrix.matrix[matrix.matrix < 0] -= 1
        # matrix.epsilon_modification()
        matrices.append(matrix.matrix)
    c1n = getattr(corpus2, "name", "Corpus 1")
    c2n = getattr(corpus1, "name", "Corpus 2")
    cdist_ = cdist(matrices[0], matrices[1], metric=distance)
    if group_by:
        reduce_df["groups"] = (
            reduce_df[group_by] != reduce_df[group_by].shift()
        ).cumsum()
        reduce_df = (
            reduce_df.reset_index()
            .groupby("groups")
            .agg({"index": list, group_by: "first"})
        )
        combs = list(cwr(reduce_df.index, 2))
        d = dict(zip([c1n, c2n], list(zip(*combs)))) | {"value": []}
        ixs = [reduce_df.loc[[i1, i2], "index"].tolist() for i1, i2 in combs]
        for ix1, ix2 in ixs:
            submatrix = cdist_[ix1][:, ix2]
            d["value"].append(submatrix.sum() / np.multiply(*submatrix.shape))
        reduce_df.loc[reduce_df.duplicated(subset=group_by), group_by] += " "
        df = pd.DataFrame(d).replace(reduce_df[group_by].to_dict())[[c2n, c1n, "value"]]
    else:
        cdist_[np.triu_indices(len(cdist_), k=1)] = np.nan
        df = pd.DataFrame(
            cdist_,
            index=corpus1.document_cat.categories,
            columns=corpus2.document_cat.categories,
        )
        if c1n == c2n:
            c2n = c1n + " "
        df = (
            df.rename_axis(index=c1n)
            .melt(ignore_index=False, var_name=c2n)
            .dropna()
            .reset_index()
        )
    df["value"] /= df["value"].max()
    if res == "matrix":
        df = df.pivot(index=c1n, columns=c2n, values="value").fillna(0)
        df = df + df.T
    return df


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

# freq = pd.read_csv("./frequency.csv")
# corpus = Corpus(freq=freq)
# matrix = corpus.pivot()
# dvr = corpus.create_dvr(matrix)
# epsilon = 1 / (len(dvr) * 2)
# create_signatures(matrix=matrix, corpus=corpus, epsilon=epsilon)[1].sum(axis=0)


def PCA(sockpuppet_matrix, n_components: int = 2):
    """
    Creates a PCA object and returns it.
    """
    scaler = StandardScaler()
    sockpuppet_matrix = scaler.fit_transform(sockpuppet_matrix)
    scaled_matrix = scaler.fit_transform(sockpuppet_matrix)
    pca = skd.PCA(n_components=n_components)
    pca.fit(scaled_matrix)
    res = pca.transform(scaled_matrix)
    return res, pca.explained_variance_ratio_
