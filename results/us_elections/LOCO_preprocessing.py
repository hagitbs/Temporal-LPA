from collections import Counter
from glob import glob
from pathlib import Path
import os
from typing import Literal

import ijson
import nltk
import pandas as pd
import simplejson as json
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.corpus import stopwords
import tomli

from helpers import read, write


if "WSL_DISTRO_NAME" not in os.environ.keys():
    nltk.download("stopwords")
    nltk.download("punkt")
    stemmer = SnowballStemmer("english")
    ignored_words = nltk.corpus.stopwords.words("english")
DATA_FOLDER = Path("./data")
BIGLOCO = "/home/alexzabbey/ossi/lpa/Temporal-LPA/data/LOCO.json"
ignore_tokens = [",", ".", ";", ":", '"', "``", "''", "`", "--", "-", "..."]
wnl = WordNetLemmatizer()
full_stopwords = set(
    stopwords.words("english")
    + [
        "would",
        "also",
        "upon",
        "must",
        "u",
        "every",
        "shall",
        "many",
        "without",
        "much",
        "within",
        "could",
        "should",
        "since",
    ]
)


def count_em(article):
    c = Counter()
    for sentence in article.split("\r\n\r\n"):
        word_list = nltk.word_tokenize(sentence)
        word_list = [
            word.lower()
            for word in word_list
            if word.isalpha() and word.lower() not in ignored_words
        ]
        stem_words = [stemmer.stem(word) for word in word_list]
        c += Counter(stem_words)
    return c


def df_it(d, i, j):
    df = (
        pd.DataFrame.from_dict(d, orient="index")
        .rename_axis(index="123456789")
        .reset_index()
        .melt(id_vars="the")
        .dropna(subset=["value"])
        .rename(
            columns={
                "variable": "element",
                "the": "document",
                "value": "frequency_in_document",
            }
        )
        .sort_values("category")
    )
    df.to_csv(f"./data/loco/np_freq/frequency_{i}.csv", index=False)


def df_it2(d, i, j):
    reform = {
        (outerKey, innerKey): values
        for outerKey, innerDict in d.items()
        for innerKey, values in innerDict.items()
    }
    tight = {
        "data": list(reform.values()),
        "index": list(reform.keys()),
        "columns": ["frequency_in_document"],
        "column_names": [None],
        "index_names": ["document", "element"],
    }
    df = pd.DataFrame.from_dict(tight, orient="tight").reset_index()
    df.to_csv(f"./data/loco/np_freq/frequency_{i}.csv", index=False)


def break_up_json():
    with open(BIGLOCO) as f:
        data = ijson.items(f, "item")
        items = [v for v in data]
        for i in range(0, len(items), 5000):
            with open(f"data/loco/json/LOCO_{i}.json", "w") as fp:
                json.dump(items[i : i + 5000], fp)


def lemmatize(s):
    wl = word_tokenize(s)
    word_list = [
        word.lower()
        for word in wl
        if word.isalpha() and word.lower() not in full_stopwords
    ]
    lemmatized_words = [wnl.lemmatize(word) for word in word_list]
    return Counter(lemmatized_words)


# def create_frequencies():
#     # FIXME: misses last 743
#     for i in range(5000, 100_000, 5000):
#         with open(f"data/loco/json/LOCO_{i}.json", "r") as f:
#             data = json.load(f)
#         d = {}
#         for j, article in enumerate(data):
#             txt = (
#                 article["txt"]
#                 .replace("‘", "'")
#                 .replace("’", "'")
#                 .replace("“", '"')
#                 .replace("”", '"')
#             )
#             txt = TextBlob(txt)
#             d[article["doc_id"]] = txt.np_counts
#             if len(d) % 1000 == 0:
#                 df_it2(d, i, j)
#                 d = {}


def create_frequencies():
    # FIXME: misses last 743
    for i in range(60000, 100_000, 5000):
        with open(f"data/loco/json/LOCO_{i}.json", "r") as f:
            data = json.load(f)
        d = {}
        for j, article in enumerate(data):
            txt = (
                article["txt"]
                .replace("‘", "'")
                .replace("’", "'")
                .replace("“", '"')
                .replace("”", '"')
            )
            txt = lemmatize(txt)
            d[article["doc_id"]] = txt
        df_it2(d, i, j)


def create_metadata():
    for sc in ("conspiracy", "mainstream"):
        with open(BIGLOCO) as f:
            data = ijson.items(f, "item")
            items = [(v["doc_id"], v.get("date", pd.NA), v["subcorpus"]) for v in data]
        metadata = pd.DataFrame(items, columns=["document", "date", "subcorpus"])
        metadata = (
            metadata[metadata["subcorpus"] == sc]
            .dropna(subset=["date"])
            .drop(columns=["subcorpus"])
        )
        metadata = metadata[metadata["date"].str[:4].astype(int) >= 2000]
        metadata.to_csv(f"data/loco_{sc}/metadata.csv", index=False)


def create_freq(metadata) -> pd.DataFrame:
    base_freq = []
    for p in glob(f"data/loco/np_freq/*.csv"):
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


# break_up_json()
# create_frequencies()
# create_metadata()
with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)
RESULTS_FOLDER = Path("results") / config["corpus"]
print(config["corpus"])
metadata = pd.read_csv(f"data/{config['corpus']}/metadata.csv")
base_freq = create_freq(metadata)
write(DATA_FOLDER / config["corpus"], base_freq, "base_freq")
print(DATA_FOLDER / config["corpus"])
base_freq = read(DATA_FOLDER / config["corpus"], "base_freq.csv")
freq = tw_freq(base_freq, config["freq"]).rename(
    columns={"date": "document", "frequency_in_category": "frequency_in_document"}
)
write(DATA_FOLDER / config["corpus"], freq, "freq")
freq = read(DATA_FOLDER / config["corpus"], f"freq.csv", parse_date="document")
print(len(freq))
# ####
# df["counter"] = df["text"].apply(lemmatize)
# df = df.set_index("year")
