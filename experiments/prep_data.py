"""Prepare the real-world datasets under data/ (not tracked).

  mnist.npz     MNIST from the local idx files (cobweb-poincare/data/MNIST/raw)
  housing.npz   California housing (Pace & Barry 1997; the handson-ml CSV), 8 numeric
                features + log median house value
  owt.npz       the first documents of OpenWebText (Skylion007/openwebtext, streamed),
                lower-cased word tokens; vocabulary = top-V words + <unk> + <s>
"""
import argparse
import csv
import io
import os
import re
import urllib.request

import numpy as np


def idx(path):
    with open(path, "rb") as fh:
        data = fh.read()
    ndim = data[3]
    shape = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def mnist(out):
    raw = "/coc/flash10/ksingara3/cobweb-poincare/data/MNIST/raw"
    np.savez_compressed(os.path.join(out, "mnist.npz"),
                        xtr=idx(f"{raw}/train-images-idx3-ubyte"), ytr=idx(f"{raw}/train-labels-idx1-ubyte"),
                        xte=idx(f"{raw}/t10k-images-idx3-ubyte"), yte=idx(f"{raw}/t10k-labels-idx1-ubyte"))


def housing(out):
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(url).read().decode())))
    feats = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
             "population", "households", "median_income"]
    X, y = [], []
    for r in rows:
        if any(r[k] == "" for k in feats):
            continue
        hh = float(r["households"])
        # the usual per-household features (as in sklearn's version of the dataset)
        X.append([float(r["longitude"]), float(r["latitude"]), float(r["housing_median_age"]),
                  float(r["total_rooms"]) / hh, float(r["total_bedrooms"]) / hh,
                  float(r["population"]) / hh, hh, float(r["median_income"])])
        y.append(np.log(float(r["median_house_value"])))
    X = np.array(X, dtype=np.float32)
    X[:, 3:7] = np.log1p(X[:, 3:7])
    np.savez_compressed(os.path.join(out, "housing.npz"), X=X, y=np.array(y, dtype=np.float32),
                        names=np.array(["lon", "lat", "age", "log_rooms", "log_bedrooms",
                                        "log_occupancy", "log_households", "income"]))


def owt(out, n_words, vocab):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    words = []
    for ex in ds:
        words.append("<s>")
        words.extend(re.findall(r"[a-z]+|[0-9]+|[.,!?;:'\"()-]", ex["text"].lower()))
        if len(words) >= n_words:
            break
    from collections import Counter
    cnt = Counter(w for w in words if w != "<s>")
    vocab_list = ["<unk>", "<s>"] + [w for w, _ in cnt.most_common(vocab)]
    index = {w: i for i, w in enumerate(vocab_list)}
    ids = np.array([index.get(w, 0) for w in words], dtype=np.int32)
    np.savez_compressed(os.path.join(out, "owt.npz"), ids=ids, vocab=np.array(vocab_list))
    print("owt words", len(ids), "unk rate", float(np.mean(ids == 0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="mnist,housing,owt")
    ap.add_argument("--owt_words", type=int, default=2_000_000)
    ap.add_argument("--vocab", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    for w in a.which.split(","):
        {"mnist": lambda: mnist("data"), "housing": lambda: housing("data"),
         "owt": lambda: owt("data", a.owt_words, a.vocab)}[w]()
        print("done", w, flush=True)
