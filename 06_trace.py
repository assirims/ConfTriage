"""Step 06 - trace link recovery: ranking quality and conformal recall control.

Scores for every requirement-code pair
  retrieval:   TF-IDF, BM25, MiniLM, BGE, cross-encoder, reciprocal-rank fusion of BM25 + BGE (the candidate generator)
  generative:  each LLM verifies the top-K candidates of every requirement (P(Yes)); candidates are ranked by it and
               placed above the remaining pairs, which keep their fusion order
  fusion:      (a) LLM + RRF: training-free reciprocal rank fusion of BM25, BGE and the LLM ranking of the shortlist;
               (b) logistic regression on within-requirement standardised retrieval scores, candidate rank and the LLM
               log-odds, trained on the OTHER data sets only (leave-one-data-set-out), so no label of the target
               system is used; 'Retrieval fusion' is the same model without the LLM feature
Ranking metrics: mean average precision (MAP) and recall@5/10 over requirements with at least one gold link, with
bootstrap CIs over requirements and paired Wilcoxon tests.
Conformal recall control (conformal risk control with requirements as exchangeable units): a score threshold
calibrated on a random half of the requirements so that the expected share of a requirement's true links that is
kept is >= 1 - alpha; the analyst vets only the kept links. Evaluated over N_SPLITS random halves, and under shift
(calibration on the other data sets).
Outputs: results/tables/trace_*.csv, data/processed/trace_scores.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LLM_LABEL, LLMS, N_BOOT, PROC, SEED, TABLES, TRACE_SETS  # noqa: E402
from lib.conformal import crc_threshold  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
N_SPLITS = int(os.environ.get("TRACE_SPLITS", 500))
TRACE_ALPHAS = (0.10, 0.20)
RETRIEVAL = {"TF-IDF": "tfidf", "BM25": "bm25", "MiniLM": "minilm", "BGE": "bge", "Cross-encoder": "cross",
             "RRF (BM25 + BGE)": "rrf"}


def zscore_within(D, col):
    g = D.groupby(["dataset", "req"])[col]
    return ((D[col] - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0.0)


def build_scores():
    D = pd.read_csv(os.path.join(PROC, "trace_pairs.csv"))
    for c in ("bm25", "bge", "cross", "tfidf"):
        D[f"z_{c}"] = zscore_within(D, c)
    D["neg_log_rank"] = -np.log(D.rrf_rank)
    models = []
    for m in LLMS:
        f = os.path.join(PROC, f"llm_trace_{m}.csv")
        if os.path.exists(f):
            L = pd.read_csv(f)[["dataset", "req", "code", "p_yes", "logit_yes"]].rename(
                columns={"p_yes": f"p_{m}", "logit_yes": f"logit_{m}"})
            D = D.merge(L, on=["dataset", "req", "code"], how="left")
            if D.loc[D.cand == 1, f"p_{m}"].isna().any():
                raise ValueError(f"incomplete LLM trace outputs for {m}")
            models.append(m)
    S = pd.DataFrame({"dataset": D.dataset, "req": D.req, "code": D.code, "gold": D.gold, "cand": D.cand})
    for name, col in RETRIEVAL.items():
        S[name] = D[col]
    g = D.groupby(["dataset", "req"])
    r_bm25 = g.bm25.rank(ascending=False, method="first")
    r_bge = g.bge.rank(ascending=False, method="first")
    for m in models:
        lab = LLM_LABEL[m]
        S[f"{lab} re-rank"] = np.where(D.cand == 1, 1.0 + D[f"p_{m}"].fillna(0), D.rrf)
        # training-free rank fusion: reciprocal ranks of BM25, BGE and the LLM judgement (shortlist only)
        r_llm = D[f"p_{m}"].where(D.cand == 1).groupby([D.dataset, D.req]).rank(ascending=False, method="first")
        r_llm = r_llm.fillna(D.groupby(["dataset", "req"]).code.transform("count"))
        S[f"{lab} + RRF"] = 1.0 / (60 + r_bm25) + 1.0 / (60 + r_bge) + 1.0 / (60 + r_llm)
    base = ["z_bm25", "z_bge", "z_cross", "z_tfidf", "neg_log_rank", "cand"]
    feats = {"Retrieval fusion": base}
    for m in models:
        D[f"cl_{m}"] = np.where(D.cand == 1, D[f"logit_{m}"].fillna(0).clip(-15, 15), 0.0)
        feats[f"{LLM_LABEL[m]} + retrieval fusion"] = base + [f"cl_{m}"]
    coefs = []
    for name, cols in feats.items():
        S[name] = np.nan
        for ds in TRACE_SETS:
            tr, te = D.dataset != ds, D.dataset == ds
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(D.loc[tr, cols], D.loc[tr, "gold"])
            S.loc[te, name] = clf.predict_proba(D.loc[te, cols])[:, 1]
            coefs.append({"method": name, "held_out": ds, **dict(zip(cols, clf.coef_[0]))})
    S.to_csv(os.path.join(PROC, "trace_scores.csv"), index=False)
    pd.DataFrame(coefs).to_csv(os.path.join(TABLES, "trace_fusion_coefficients.csv"), index=False)
    return S, [c for c in S.columns if c not in ("dataset", "req", "code", "gold", "cand")]


def per_requirement(S, ds, method):
    """Average precision, recall@5 and recall@10 of every requirement of `ds` with at least one gold link."""
    out = []
    for r, g in S[S.dataset == ds].groupby("req"):
        if g.gold.sum() == 0:
            continue
        o = g.sort_values(method, ascending=False, kind="stable").gold.values
        hits = np.cumsum(o)
        prec = hits / np.arange(1, len(o) + 1)
        out.append({"req": r, "ap": float((prec * o).sum() / o.sum()), "r5": hits[min(4, len(o) - 1)] / o.sum(),
                    "r10": hits[min(9, len(o) - 1)] / o.sum()})
    return pd.DataFrame(out)


def boot_ci(x, rng, B=N_BOOT):
    x = np.asarray(x, float)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return x.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def ranking_tables(S, methods):
    rng = np.random.default_rng(SEED)
    rows, per = [], {}
    for ds in TRACE_SETS:
        for m in methods:
            pr = per_requirement(S, ds, m)
            per[(ds, m)] = pr.set_index("req")
            row = {"dataset": ds, "method": m, "n_req": len(pr)}
            for k in ("ap", "r5", "r10"):
                e, lo, hi = boot_ci(pr[k], rng)
                row.update({k: e, f"{k}_lo": lo, f"{k}_hi": hi})
            rows.append(row)
    T = pd.DataFrame(rows).rename(columns={"ap": "MAP", "ap_lo": "MAP_lo", "ap_hi": "MAP_hi"})
    tests = []
    for ds in TRACE_SETS:
        for m in methods:
            for ref in ("RRF (BM25 + BGE)", "Cross-encoder", "Retrieval fusion"):
                if m == ref or not (m.endswith("re-rank") or m.endswith("+ retrieval fusion") or m.endswith("+ RRF")):
                    continue
                a, b = per[(ds, m)].ap, per[(ds, ref)].ap
                d = (a - b).values
                p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
                e, lo, hi = boot_ci(d, rng)
                tests.append({"dataset": ds, "method": m, "vs": ref, "map_diff": e, "lo": lo, "hi": hi, "p_wilcoxon": p})
    return T, pd.DataFrame(tests)


def recall_loss_matrix(G, scores_by_req, lambdas):
    """loss[i, j] = 1 - share of requirement i's gold links with score >= lambdas[j] (lambdas in decreasing order)."""
    L = np.zeros((len(G), len(lambdas)))
    for i, (s, g) in enumerate(zip(scores_by_req, G)):
        gs = np.sort(s[g == 1])
        kept = len(gs) - np.searchsorted(gs, lambdas, side="left")
        L[i] = 1.0 - kept / len(gs)
    return L


def kept_matrix(scores_by_req, lambdas):
    return np.array([len(s) - np.searchsorted(np.sort(s), lambdas, side="left") for s in scores_by_req])


def crc_eval(S, methods, universe="all"):
    """universe='all': every requirement-code pair (recall of all gold links); universe='shortlist': only the top-K
    candidates of each requirement (the analyst vets a pruned shortlist; recall of the gold links in the shortlist)."""
    rng = np.random.default_rng(SEED + 7)
    rows, curves = [], []
    for ds in TRACE_SETS:
        Sd = S[S.dataset == ds]
        if universe == "shortlist":
            Sd = Sd[Sd.cand == 1]
        groups = [g for _, g in Sd.groupby("req")]
        has = np.array([g.gold.sum() > 0 for g in groups])
        n_code = Sd.code.nunique()
        for m in methods:
            sc = [g[m].values for g in groups]
            gl = [g.gold.values for g in groups]
            lambdas = np.unique(np.quantile(Sd[m].values, np.linspace(0, 1, 801)))[::-1]
            lambdas = np.concatenate([[np.inf], lambdas])
            Lfull = np.zeros((len(groups), len(lambdas)))
            Lfull[has] = recall_loss_matrix([gl[i] for i in np.where(has)[0]], [sc[i] for i in np.where(has)[0]], lambdas)
            Kept = kept_matrix(sc, lambdas)
            Gold = np.array([g.sum() for g in gl])
            # trade-off curve (all requirements): mean recall vs links kept per requirement
            for j in range(0, len(lambdas), 8):
                curves.append({"universe": universe, "dataset": ds, "method": m, "recall": 1 - Lfull[has, j].mean(),
                               "kept_per_req": Kept[:, j].mean()})
            for a in TRACE_ALPHAS:
                rec, eff, prec, ok = [], [], [], []
                rng = np.random.default_rng(SEED + 7)          # identical calibration/test splits for every method
                idx = np.arange(len(groups))
                for _ in range(N_SPLITS):
                    rng.shuffle(idx)
                    cal, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
                    cal_h = cal[has[cal]]
                    j = crc_threshold(Lfull[cal_h], lambdas, a)
                    te_h = te[has[te]]
                    r = 1 - Lfull[te_h, j].mean()
                    rec.append(r)
                    ok.append(r >= 1 - a)
                    eff.append(Kept[te, j].mean())
                    tp = sum(((sc[i] >= lambdas[j]) & (gl[i] == 1)).sum() for i in te)
                    prec.append(tp / max(1, Kept[te, j].sum()))
                rows.append({"universe": universe, "dataset": ds, "method": m, "alpha": a, "target_recall": 1 - a, "recall_mean": np.mean(rec),
                             "recall_sd": np.std(rec), "share_splits_met": np.mean(ok), "kept_per_req": np.mean(eff),
                             "kept_per_req_sd": np.std(eff), "share_of_pairs": np.mean(eff) / n_code,
                             "precision": np.mean(prec), "n_req": len(groups), "n_code": n_code,
                             "gold_per_req": Gold.mean(), "n_splits": N_SPLITS})
        print(f"CRC {universe} {ds} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(curves)


def crc_shift(S, methods):
    """Calibrate on the requirements of the other data sets (no labels of the target system), test on the target.
    Only scores that are not trained on any data set are compared, so calibration data never informed the scores."""
    rows = []
    for m in methods:
        for ds in TRACE_SETS:
            cal, te = S[S.dataset != ds], S[S.dataset == ds]
            lambdas = np.concatenate([[np.inf], np.unique(np.quantile(S[m].values, np.linspace(0, 1, 801)))[::-1]])

            def mats(X):
                gr = [g for _, g in X.groupby(["dataset", "req"]) if g.gold.sum() > 0]
                return recall_loss_matrix([g.gold.values for g in gr], [g[m].values for g in gr], lambdas)
            for a in TRACE_ALPHAS:
                j = crc_threshold(mats(cal), lambdas, a)
                rec = 1 - mats(te)[:, j].mean()
                kept = np.mean([(g[m].values >= lambdas[j]).sum() for _, g in te.groupby("req")])
                rows.append({"method": m, "dataset": ds, "alpha": a, "target_recall": 1 - a, "recall": rec,
                             "met": rec >= 1 - a, "kept_per_req": kept})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    S, methods = build_scores()
    print("methods:", methods, flush=True)
    T, tests = ranking_tables(S, methods)
    T.to_csv(os.path.join(TABLES, "trace_ranking.csv"), index=False)
    tests.to_csv(os.path.join(TABLES, "trace_ranking_tests.csv"), index=False)
    print(T[["dataset", "method", "MAP", "r5", "r10"]].round(3).to_string(), flush=True)
    C1, cu1 = crc_eval(S, methods, "all")
    C2, cu2 = crc_eval(S, methods, "shortlist")
    C = pd.concat([C1, C2], ignore_index=True)
    C.to_csv(os.path.join(TABLES, "trace_crc.csv"), index=False)
    pd.concat([cu1, cu2], ignore_index=True).to_csv(os.path.join(TABLES, "trace_tradeoff_curves.csv"), index=False)
    # ranking quality inside the shortlist (how well each score orders the K candidates)
    Ts, _ = ranking_tables(S[S.cand == 1], methods)
    Ts.to_csv(os.path.join(TABLES, "trace_ranking_shortlist.csv"), index=False)
    print(Ts[["dataset", "method", "MAP", "r5", "r10"]].round(3).to_string(), flush=True)
    untrained = [m for m in methods if "fusion" not in m]
    crc_shift(S, untrained).to_csv(os.path.join(TABLES, "trace_crc_shift.csv"), index=False)
    print(C[C.alpha == 0.1][["universe", "dataset", "method", "recall_mean", "share_splits_met", "kept_per_req", "precision"]].round(3).to_string())
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
