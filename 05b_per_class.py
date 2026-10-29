"""Step 05b - per-class F1 and recall: each LLM configuration (zero-shot, all evaluation requirements) and the
lightweight classifiers (out-of-fold predictions, 4 folds grouped by project)."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, LLM_LABEL, LLMS, PROC, TABLES  # noqa: E402


def main():
    P = pd.read_csv(os.path.join(PROC, "promise_exp.csv"))
    keep = (P.exemplar == 0).values
    E = {k: np.load(os.path.join(PROC, f"emb_promise_{k}.npy"))[keep] for k in ("minilm", "bge")}
    P = P[keep].reset_index(drop=True)
    y = P.label.map(CLASSES.index).values
    preds = {}
    for name, key in (("MiniLM + LR", "minilm"), ("BGE + LR", "bge"), ("TF-IDF + LR", None)):
        oof = np.zeros(len(y), int)
        for a, b in GroupKFold(4).split(P, groups=P.project):
            if key:
                Xa, Xb = E[key][a], E[key][b]
            else:
                tf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
                Xa, Xb = tf.fit_transform(P.text.values[a]), tf.transform(P.text.values[b])
            oof[b] = LogisticRegression(C=10.0, max_iter=1000, tol=1e-3).fit(Xa, y[a]).predict(Xb)
        preds[name + " (out-of-fold)"] = oof
    for m in LLMS:
        L = pd.read_csv(os.path.join(PROC, f"llm_cls_{m}.csv"))
        L = L[L["mode"] == "zero"].set_index("id").reindex(P.id)
        preds[f"{LLM_LABEL[m]} zero-shot"] = L[[f"p_{c}" for c in CLASSES]].values.argmax(1)
    rows = []
    for name, p in preds.items():
        f1 = f1_score(y, p, labels=range(len(CLASSES)), average=None, zero_division=0)
        rc = recall_score(y, p, labels=range(len(CLASSES)), average=None, zero_division=0)
        for k, c in enumerate(CLASSES):
            rows.append({"method": name, "class": c, "n": int((y == k).sum()), "f1": f1[k], "recall": rc[k]})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "cls_per_class.csv"), index=False)
    print(pd.DataFrame(rows).pivot(index="method", columns="class", values="f1")[CLASSES].round(2).to_string())


if __name__ == "__main__":
    main()
