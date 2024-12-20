from collections import Counter
from pathlib import Path
import os 
import ijson 
import nltk
import pandas as pd
import simplejson as json
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob

if "WSL_DISTRO_NAME" not in os.environ.keys():
    nltk.download("stopwords")
    nltk.download("punkt")
    stemmer = SnowballStemmer("english")
    ignored_words = nltk.corpus.stopwords.words("english")
DATA_FOLDER = Path("./data")


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
                "the": "category",
                "value": "frequency_in_category",
            }
        )
        .sort_values("category")
    )
    df.to_csv(f"./data/loco/np_freq/frequency_{(i + j + 1)-1000}.csv", index=False)


def df_it2(d, i, j):
    reform = {
        (outerKey, innerKey): values
        for outerKey, innerDict in d.items()
        for innerKey, values in innerDict.items()
    }
    tight = {
        "data": list(reform.values()),
        "index": list(reform.keys()),
        "columns": ["frequency_in_category"],
        "column_names": [None],
        "index_names": ["category", "element"],
    }
    df = pd.DataFrame.from_dict(tight, orient="tight").reset_index()
    df.to_csv(f"./data/loco/np_freq/frequency_{(i + j + 1)-1000}.csv", index=False)


def break_up_json():
    with open("data/loco/json/LOCO.json") as f:
        data = ijson.items(f, "item")
        items = [v for v in data]
        for i in range(0, len(items), 5000):
            with open(f"data/loco/json/LOCO_{i}.json", "w") as fp:
                json.dump(items[i : i + 5000], fp)


def create_frequencies():
    # FIXME: misses last 743
    for i in range(5000, 100_000, 5000):
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
            txt = TextBlob(txt)
            d[article["doc_id"]] = txt.np_counts
            if len(d) % 1000 == 0:
                df_it2(d, i, j)
                d = {}


def create_metadata():
    with open ("./data/loco/json/LOCO.json")  as f:  
        data = ijson.items(f, "item")
        items = [(v["doc_id"], v.get("date", pd.NA), v["subcorpus"]) for v in data]
    metadata = pd.DataFrame(items, columns=["category", "date", "subcorpus"])
    for sc in ("conspiracy", "mainstream"):
        metadata[metadata["subcorpus"] == sc].drop(columns=["subcorpus"]).to_csv(
            f"data/loco_{sc}/metadata.csv", index=False
        )


# break_up_json()
# create_frequencies()
create_metadata()