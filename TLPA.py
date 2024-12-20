from glob import glob
from pathlib import Path
from typing import List, Literal, Tuple
from math import floor

import pandas as pd
import tomli
from more_itertools import consecutive_groups

from LPA import Corpus, Matrix, sockpuppet_distance
from algo import symmetrized_KLD
from helpers import read, write
from visualize import metric_bar_chart, moving_avg, sockpuppet_matrix

with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)
PATH = Path("results") / config["corpus"]


def create_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata = read(Path("data") / config["corpus"], "metadata.csv")
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


# def create_freq(threshold=20) -> pd.DataFrame:
#     metadata, filter_ = create_metadata(threshold)
#     base_freq = []
#     for i in range(0, 96000, 1000):
#         base_freq.append(pd.read_csv(f"data/loco/np_freq/frequency_{i}.csv"))
#     base_freq = pd.merge(
#         pd.concat(base_freq),
#         metadata[["category", "date", "subcorpus"]],
#         on="category",
#         how="inner",
#     ).rename(columns={"count": "frequency_in_category"})
#     base_freq["dt"] = base_freq["date"].astype("str").str[:7]

#     base_freq = base_freq.set_index(["subcorpus", "dt"])
#     base_freq = (
#         base_freq.loc[filter_.index]
#         .sort_values("date")
#         .reset_index(level=0)
#         .reset_index(drop=True)
#     )
#     return base_freq


def create_freq() -> pd.DataFrame:
    metadata = create_metadata()
    print(metadata.head())
    raise
    for sc in ("conspiracy", "mainstream"):
        base_freq = []
        for p in glob(f"data/{config['corpus']}/np_freq/*.csv"):
            base_freq.append(pd.read_csv(p))
        base_freq = pd.merge(
            pd.concat(base_freq),
            metadata[["document", "date"]],
            on="document",
            how="inner",
        ).rename(columns={"count": "frequency_in_document"})
        base_freq["document"] = pd.Categorical(base_freq["document"])
        return base_freq


def tw_freq(
    base_freq: pd.DataFrame, freq: Literal["MS", "D", "W"] = "MS"
) -> pd.DataFrame:
    df = base_freq.groupby([pd.Grouper(freq=freq, key="date"), "element"]).sum(
        numeric_only=True
    )
    df = df.reset_index()  # .rename(
    #     columns={"frequency_in_category": "frequency_in_document"}
    # )

    # res = (
    #     (df / base_freq.resample(freq, on="date").sum(numeric_only=True))
    #     .reset_index()
    #     .rename(columns={"frequency_in_category": "global_weight"})
    # )
    return df


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
    x = symmetrized_KLD(dvr, matrix.matrix)
    distances = pd.DataFrame(
        x,
        index=corpus.date_cat.categories,
        columns=corpus.element_cat.categories,
    )
    max_distances = distances[distances.abs().sum().nlargest(30).index]
    distances = [sig.sort_values(ascending=False) for _, sig in distances.iterrows()]
    # distances = distances.stack().reset_index().rename(columns={0: "KL"})
    return distances, max_distances


# def element_kv():
#     k = ["שם יישוב", "סמל יישוב"]
#     v = ["name", "element"]
#     kv = pd.read_excel("data/elections/bycode2021.xlsx")[k].rename(
#         columns=dict(zip(k, v))
#     )
#     return pd.concat(
#         [kv, pd.DataFrame(["מעטפות חיצוניות", 99999], index=v).T], ignore_index=True
#     )


if __name__ == "__main__":
    """
    The main process of this code reads
    """
    base_freq = create_freq()
    write(PATH, base_freq, "base_freq")
    # # # base_freq = read("base_freq.csv")
    tw_freq_df = tw_freq(base_freq, config["freq"])
    write(PATH, tw_freq_df, "tw_freq")
    # # tw_freq_df = read(PATH, f"tw_freq.csv")

    # ####
    freq = tw_freq_df.rename(
        columns={"date": "document", "frequency_in_category": "frequency_in_document"}
    )
    # freq[freq["document"] == pd.to_datetime("2011-05-01")].sort_values(
    #     "frequency_in_document", ascending=False
    # ).to_csv("may2011mains.csv", index=False)
    # freq = pd.read_csv(f"data/{config['corpus']}/np_freq/1.csv")
    corpus = Corpus(freq=freq, name=config["corpus"])
    dvr = corpus.create_dvr(equally_weighted=True)
    write(PATH, dvr, "dvr")
    # epsilon_frac = 2
    # epsilon = 1 / (len(dvr) * epsilon_frac)
    epsilon = config["epsilon"]
    # epsilon = 4.07880630809523e-07
    # print(Process().memory_info().rss)
    prevelent = floor(len(corpus) * config["prevelent"])
    signatures, most_significant, temporary_array = corpus.create_signatures(
        epsilon=epsilon,
        most_significant=30,
        sig_length=config["sig_length"],
        prevelent=prevelent,
    )
    spd = sockpuppet_distance(corpus, corpus)

    write(PATH, temporary_array, "temporary_array")
    # print(Process().memory_info().rss)
    average_distance = corpus.create_dvr(
        equally_weighted=True, matrix=corpus.distance_matrix
    )
    # write(PATH, average_distance, "average_distance", color=True)
    # write(PATH, corpus.matrix, "matrix")
    # dvr = read(PATH, "dvr.csv")
    # matrix = Matrix(read(PATH, "matrix.npy"))
    # sigs, max_distances = create_distances(matrix, dvr, corpus)
    # write(PATH, most_significant, "max_distances")
    # wordlist = {}
    # word = config["word"]
    for sig in signatures:
        name = sig.name
        sig = sig.rename("KL").reset_index()
        # wordlist[name] = sig[sig["index"] == word].loc[:, "KL"].iloc[0]
        write(
            PATH,
            sig,
            f"sigs/sigs_with_prevelence_{prevelent}_{name}_epsilon_{epsilon}",
            color=True,
        )
        # symmetrized_KLD(sig, average_distance)
    # word_df = (
    #     pd.DataFrame.from_dict(wordlist, orient="index")
    #     .rename(columns={0: word})
    #     .sort_index()
    # )
    # print(word_df.mean())
    # write(PATH, word_df, f"sigs/{word}_{epsilon}", color=True)
    spname = f"republican_sockpuppets.html"
    # sockpuppet_matrix(spd).save(PATH / spname)
    # for df in most_significant:
    #     name = sig.name.strftime("%Y-%m-%d")
    #     write(PATH, (sig.rename("KL").reset_index(), f"most_significant/ms_{name}"))
