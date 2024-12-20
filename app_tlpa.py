from glob import glob
from pathlib import Path
from typing import List, Literal, Tuple

import pandas as pd
import tomli
from more_itertools import consecutive_groups
from streamlit.runtime.uploaded_file_manager import UploadedFile
from corpora import Corpus, Matrix
from algo import KLD
from helpers import read, write
from visualize import metric_bar_chart

with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)
PATH = Path("results") / config["corpus"]


def create_metadata(file_like: UploadedFile) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(file_like, parse_dates=["date"])
    metadata = metadata[
        (metadata["date"] >= pd.Timestamp(config["start_date"]))
        & (metadata["date"] <= pd.Timestamp(config["end_date"]))
    ]
    if config["threshold"] > 0:
        resampled = metadata.resample(config["freq"], on="date").count()
        filter_ = resampled[resampled > config["threshold"]].dropna()
        filter_ = filter_.index.astype(str).str[:7]
        metadata = metadata[metadata["date"].astype(str).str[:7].isin(filter_)]
    return metadata


def create_freq(metadata: UploadedFile, data: List[UploadedFile]) -> pd.DataFrame:
    metadata = create_metadata(metadata)
    base_freq = pd.concat([pd.read_csv(f) for f in data])
    base_freq = pd.merge(metadata[["category", "date"]], on="category", how="inner")
    base_freq["category"] = pd.Categorical(base_freq["category"])
    return base_freq


def tw_freq(
    base_freq: pd.DataFrame, freq: Literal["MS", "D", "W"] = "MS"
) -> pd.DataFrame:
    df = base_freq.groupby([pd.Grouper(freq=freq, key="date"), "element"]).sum()
    res = (
        (df / base_freq.resample(freq, on="date").sum())
        .reset_index()
        .rename(columns={"frequency_in_category": "global_weight"})
    )
    return res


def check_metric(
    matrix: Matrix, delta: str | float, iter_: int
) -> List[Tuple[int, int]]:
    metric = config["metric"]
    df = matrix.apply(metric)
    if delta == "median":
        delta = df[metric].median()
    metric_bar_chart(df, rule_value=delta, metric=metric).save(
        f"results/{config['corpus']}/bar_charts/{metric}_delta_{delta}_iter_{iter_}.html"
    )
    low = df[df[metric] < delta]
    groups = [
        (min(i), max(i))
        for i in [list(x) for x in consecutive_groups(low.index)]
        if len(i) > 1
    ]
    return groups


def dBTC(
    base_freq: pd.DataFrame,
    matrix: Matrix,
    corpus: Corpus,
    delta: float | str,
) -> Tuple[Matrix, Corpus]:
    """
    δ-bounded timeline compression using Kullback-Leibler divergence under a delta threshold - in this case the median.
    """
    iter_ = 0
    groups = check_metric(matrix, delta, iter_)
    print(f"should be around {sum(b-a for a, b in groups)} iterations")
    while len(groups) > 0:
        iter_ += 1
        date_code = groups[0][0]
        date = corpus.code_to_cat(date_code)
        next_date = corpus.code_to_cat(date_code + 1)
        matrix.delete(date_code + 1, 0)
        corpus.update_dates(corpus.code_to_cat(date_code + 1))
        squeezed_matrix = squeeze_freq(base_freq, date, next_date, corpus)
        matrix.matrix[date_code] = squeezed_matrix.epsilon_modification(
            epsilon=config["epsilon"], lambda_=config["lambda"]
        )
        groups = check_metric(matrix, delta, iter_)
        print(f"finished iteration {iter_}")
        if len(groups) == 0:
            return matrix, corpus


def create_mdvr(matrix: Matrix, corpus: Corpus) -> pd.DataFrame:
    dvr = pd.DataFrame(
        {
            "element": corpus.element_cat.categories,
            "global_weight": matrix.normalized_average_weight(),
        }
    )
    dvr = (
        dvr.reset_index()
        .rename(columns={"index": "element_code"})
        .sort_values("global_weight", ascending=False)
        .reset_index(drop=True)
    )
    return dvr


def squeeze_freq(
    base_freq: pd.DataFrame,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    corpus: Corpus,
) -> Matrix:
    squeezed = (
        base_freq[(base_freq["date"] >= min_date) & (base_freq["date"] <= max_date)]
        .groupby("element", as_index=False)
        .agg({"frequency_in_category": "sum"})
    )
    squeezed["date"] = min_date
    squeezed["global_weight"] = squeezed["frequency_in_category"] / sum(
        squeezed["frequency_in_category"]
    )
    matrix = corpus.pivot(squeezed)
    return matrix


def create_distances(
    matrix: Matrix, dvr: pd.DataFrame, corpus: Corpus
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    dvr = dvr.sort_values("element_code")["global_weight"].to_numpy()
    x = KLD(dvr, matrix.matrix)
    distances = pd.DataFrame(
        x,
        index=corpus.date_cat.categories,
        columns=corpus.element_cat.categories,
    )
    max_distances = distances[distances.abs().sum().nlargest(30).index]
    distances = [sig.sort_values(ascending=False) for _, sig in distances.iterrows()]
    # distances = distances.stack().reset_index().rename(columns={0: "KL"})
    return distances, max_distances


if __name__ == "__main__":
    """
    The main process of this code reads
    """
    # base_freq = create_freq()
    # # write(PATH, (base_freq, "base_freq"))
    # # base_freq = read("base_freq.csv")
    # tw_freq_df = tw_freq(base_freq, config["freq"])
    # write((tw_freq_df, "tw_freq"))
    tw_freq_df = read(PATH, f"tw_freq.csv")
    corpus = Corpus(tw_freq_df["date"], tw_freq_df["element"])
    # matrix = corpus.pivot(tw_freq_df)
    # # moving_average(matrix, window=3)
    # matrix = matrix.epsilon_modification(
    #     epsilon=config["epsilon"], lambda_=config["lambda"]
    # )
    # check_metric(matrix, "median", 1)
    # matrix, corpus = dBTC(base_freq, matrix, vectorizor, delta="median")
    # dvr = create_mdvr(matrix, corpus)
    # write(PATH, (dvr, "dvr"))
    # write(PATH, (matrix, "matrix"))
    dvr = read(PATH, "dvr.csv")
    matrix = Matrix(read(PATH, "matrix.npy"))
    sigs, max_distances = create_distances(matrix, dvr, corpus)
    write(PATH, (max_distances, "max_distances"))
    for sig in sigs:
        name = sig.name.strftime("%Y-%m-%d")
        write(PATH, (sig.rename("KL").reset_index(), f"sigs/sigs_{name}"))
