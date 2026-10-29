"""Step 03 - sentence embeddings and retrieval scores (no generative model).

Classification: MiniLM and BGE embeddings of every PROMISE_exp requirement (inputs of the supervised baselines).
Traceability: for every requirement-code pair of each data set, lexical (TF-IDF, BM25), dense (MiniLM, BGE) and
cross-encoder scores, plus the reciprocal-rank-fusion (BM25 + BGE) candidate list whose top-K pairs are shown to
the generative models.
Outputs: data/processed/emb_promise_{minilm,bge}.npy, data/processed/trace_pairs.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BGE, BGE_QUERY_PREFIX, CROSS, MINILM, PROC, TRACE_K  # noqa: E402
from lib.encoders import BiEncoder, CrossEncoder  # noqa: E402
from lib.lexical import BM25Index, TfidfIndex  # noqa: E402


def rank_desc(x):
    """1 = highest score (ties broken by order)."""
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), dtype=int)
    r[order] = np.arange(1, len(x) + 1)
    return r


def main():
    t0 = time.time()
    mini = BiEncoder(MINILM, pooling="mean", max_len=256)
    bge = BiEncoder(BGE, pooling="cls", max_len=256, query_prefix=BGE_QUERY_PREFIX)
    P = pd.read_csv(os.path.join(PROC, "promise_exp.csv"))
    np.save(os.path.join(PROC, "emb_promise_minilm.npy"), mini.encode(P.text.tolist()))
    np.save(os.path.join(PROC, "emb_promise_bge.npy"), bge.encode(P.text.tolist()))
    print(f"PROMISE_exp embeddings done ({time.time() - t0:.0f}s)", flush=True)

    A = pd.read_csv(os.path.join(PROC, "trace_artifacts.csv"))
    L = pd.read_csv(os.path.join(PROC, "trace_links.csv"))
    gold = set(zip(L.dataset, L.req, L.code))
    cross = CrossEncoder(CROSS, max_len=256)
    rows = []
    for ds in A.dataset.unique():
        R = A[(A.dataset == ds) & (A.kind == "req")].reset_index(drop=True)
        C = A[(A.dataset == ds) & (A.kind == "code")].reset_index(drop=True)
        tf = TfidfIndex(C.text.tolist(), ngram=(1, 1)).scores(R.text.tolist())
        bm = BM25Index(C.text.tolist()).scores(R.text.tolist())
        rq_m, cd_m = mini.encode(R.summary.tolist()), mini.encode(C.summary.tolist())
        rq_b, cd_b = bge.encode(R.summary.tolist(), is_query=True), bge.encode(C.summary.tolist())
        sm, sb = rq_m @ cd_m.T, rq_b @ cd_b.T
        qs = [r for r in R.summary for _ in range(len(C))]
        ps = list(C.summary) * len(R)
        ce = cross.score(qs, ps).reshape(len(R), len(C))
        for i, r in R.iterrows():
            rb, rd = rank_desc(bm[i]), rank_desc(sb[i])
            rrf = 1.0 / (60 + rb) + 1.0 / (60 + rd)
            rr = rank_desc(rrf)
            k = min(TRACE_K, len(C))
            for j, c in C.iterrows():
                rows.append({"dataset": ds, "req": r.aid, "code": c.aid, "gold": int((ds, r.aid, c.aid) in gold),
                             "tfidf": tf[i, j], "bm25": bm[i, j], "minilm": sm[i, j], "bge": sb[i, j], "cross": ce[i, j],
                             "rrf": rrf[j], "rrf_rank": int(rr[j]), "cand": int(rr[j] <= k)})
        print(f"{ds}: {len(R)} x {len(C)} pairs scored ({time.time() - t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(PROC, "trace_pairs.csv"), index=False)
    s = D.groupby("dataset").apply(lambda g: pd.Series({"pairs": len(g), "links": g.gold.sum(), "cand_pairs": g.cand.sum(),
                                                        "links_in_cand": g[g.cand == 1].gold.sum()}), include_groups=False)
    s["cand_recall"] = s.links_in_cand / s.links
    print(s)


if __name__ == "__main__":
    main()
