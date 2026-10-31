"""Step 08 - collect every number quoted in the manuscript into results/paper_numbers.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASS_NAMES as CLASS_NAME, LLM_LABEL, LLMS, PROC, RESULTS, TABLES, TRACE_K  # noqa: E402

DOMAIN = {"eTour": ("Tourism guide", "Use cases → Java classes"), "iTrust": ("Health records", "Use cases → Java classes"),
          "SMOS": ("School monitoring", "Use cases → Java classes"), "LibEST": ("Security protocol (EST)", "Requirements → C files")}


def records(df):
    return json.loads(df.to_json(orient="records"))


def table(name):
    f = os.path.join(TABLES, name)
    try:
        return records(pd.read_csv(f))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []


def main():
    P = pd.read_csv(os.path.join(PROC, "promise_exp.csv"))
    A = pd.read_csv(os.path.join(PROC, "trace_artifacts.csv"))
    L = pd.read_csv(os.path.join(PROC, "trace_links.csv"))
    D = pd.read_csv(os.path.join(PROC, "trace_pairs.csv"))
    ev = P[P.exemplar == 0]
    N = {"data": {"cls_total": len(P), "cls_n": len(ev), "cls_projects": int(P.project.nunique()),
                  "cls_counts": P.label.value_counts().to_dict(), "cls_share_F": float((ev.label == "F").mean()),
                  "trace_reqs": int((A.kind == "req").sum()), "trace_code": int((A.kind == "code").sum()),
                  "trace_links": len(L), "trace_k": TRACE_K, "trace_pairs": len(D), "trace_cand_pairs": int(D.cand.sum())}}
    ts = []
    for ds in ["eTour", "iTrust", "SMOS", "LibEST"]:
        g = D[D.dataset == ds]
        per = g.groupby("req").gold.sum()
        ts.append({"dataset": ds, "domain": DOMAIN[ds][0], "kind": DOMAIN[ds][1], "n_req": int(g.req.nunique()),
                   "n_code": int(g.code.nunique()), "links": int(g.gold.sum()), "req_with_links": int((per > 0).sum()),
                   "links_per_req": float(per[per > 0].mean()), "cand_recall": float(g[g.cand == 1].gold.sum() / g.gold.sum())})
    N["trace_sets"] = ts
    N["models"] = [LLM_LABEL[m] for m in LLMS if os.path.exists(os.path.join(PROC, f"llm_cls_{m}.csv"))]
    # classification
    f = os.path.join(TABLES, "cls_summary.csv")
    if os.path.exists(f):
        S = pd.read_csv(f)
        N["cls_summary"] = records(S)
        N["cls_splits"] = int(pd.read_csv(os.path.join(PROC, "cls_split_results.csv")).split.max() + 1)
        N["cls_full"] = table("cls_llm_full_sample.csv")
        N["cls_mcnemar"] = table("cls_llm_mcnemar.csv")
        N["cls_paired"] = table("cls_paired_comparisons.csv")
        N["cls_class_cov"] = table("cls_class_coverage.csv")
        N["cls_class_conditional"] = table("cls_class_conditional.csv")                     # Table 14
        N["cls_class_conditional_per_class"] = table("cls_class_conditional_per_class.csv")  # Fig. 7
    N["cls_per_class"] = table("cls_per_class.csv")
    f = os.path.join(TABLES, "cls_calibration_size.csv")
    if os.path.exists(f):
        N["cls_calsize"] = table("cls_calibration_size.csv")
    # most frequent predicted class of every LLM configuration (all evaluation requirements)
    tp = {}
    for m in LLMS:
        f = os.path.join(PROC, f"llm_cls_{m}.csv")
        if os.path.exists(f):
            x = pd.read_csv(f)
            x = x[x.id.isin(ev.id)]
            for mode, g in x.groupby("mode"):
                vc = g.pred.value_counts(normalize=True)
                tp[f"{LLM_LABEL[m]}|{mode}"] = {"label": CLASS_NAME[vc.index[0]].lower(), "code": vc.index[0], "share": float(vc.iloc[0])}
    N["top_pred"] = tp
    # tracing
    f = os.path.join(TABLES, "trace_crc.csv")
    if os.path.exists(f):
        N["trace_ranking"] = table("trace_ranking.csv")
        N["trace_ranking_shortlist"] = table("trace_ranking_shortlist.csv")
        N["trace_tests"] = table("trace_ranking_tests.csv")
        N["trace_crc"] = records(pd.read_csv(f))
        N["trace_shift"] = table("trace_crc_shift.csv")
        N["trace_calsize"] = table("trace_crc_calsize.csv")                                 # Table 16, Fig. 8
        N["trace_splits"] = int(pd.read_csv(f).n_splits.iloc[0]) if "n_splits" in pd.read_csv(f).columns else int(os.environ.get("TRACE_SPLITS", 500))
    # timing of the generative models
    tm = []
    for m in LLMS:
        for task in ("cls", "trace"):
            f = os.path.join(PROC, f"llm_{task}_{m}.csv")
            if os.path.exists(f):
                x = pd.read_csv(f)
                tm.append({"model": LLM_LABEL[m], "task": task, "n": len(x), "mean_s": float(x.seconds.mean()),
                           "median_s": float(x.seconds.median())})
    N["timing"] = tm
    with open(os.path.join(RESULTS, "paper_numbers.json"), "w") as fh:
        json.dump(N, fh, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("wrote results/paper_numbers.json")


if __name__ == "__main__":
    main()
