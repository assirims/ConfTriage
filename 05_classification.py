"""Step 05 - requirements classification: accuracy, conformal prediction sets and risk-controlled automation.

Methods (probabilistic classifiers over the 12 PROMISE_exp classes)
  supervised:  TF-IDF + logistic regression, MiniLM + LR, BGE + LR (trained on the training split)
  generative:  each LLM zero-shot and few-shot (no training; probabilities from the answer-letter log-probabilities)
  fusion:      a stacking logistic regression on [log p(LLM zero-shot), log p(BGE + LR, out-of-fold)] (one per LLM)
Evaluation design: repeated random splits into training (50%), calibration (25%) and test (25%) requirements, either
by project (whole projects are held out; realistic for a new project) or by requirement (i.i.d.); N_SPLITS each.
For every split and method we record test accuracy and macro-F1 (12 classes and functional vs non-functional),
split-conformal LAC prediction sets (marginal and class-conditional/Mondrian) and Learn-then-Test automation, i.e.
the share of test requirements that can be labelled automatically with a guaranteed error rate.
Outputs: results/tables/cls_*.csv, data/processed/cls_split_results.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ALPHAS, CLASSES, DELTA, LLM_LABEL, LLMS, PROC, SEED, TABLES  # noqa: E402
from lib.conformal import crc_automation_threshold, lac_sets, lac_threshold, ltt_threshold, mondrian_sets, mondrian_thresholds  # noqa: E402
from lib.metrics import cluster_bootstrap, mcnemar_exact  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
N_SPLITS = int(os.environ.get("CLS_SPLITS", 100))
K = len(CLASSES)
EPS12 = (0.10, 0.20)        # LTT: target error rate among automatically labelled requirements (12 classes)
EPS2 = (0.05, 0.10)         # LTT: same for functional / non-functional
BUDGET = (0.02, 0.05, 0.10)  # CRC: target share of ALL requirements that are labelled automatically and wrongly
CAL_SIZES = (30, 60, 120, 10_000)   # calibration requirements (10_000 = the whole calibration split)
EPS = 1e-6
MODE_LABEL = {"zero": "zero-shot", "few": "few-shot", "zero+cc": "zero-shot + calibration", "few+cc": "few-shot + calibration"}


def load():
    P = pd.read_csv(os.path.join(PROC, "promise_exp.csv"))
    keep = (P.exemplar == 0).values
    E = {"minilm": np.load(os.path.join(PROC, "emb_promise_minilm.npy"))[keep],
         "bge": np.load(os.path.join(PROC, "emb_promise_bge.npy"))[keep]}
    P = P[keep].reset_index(drop=True)
    y = P.label.map(CLASSES.index).values
    llm = {}
    for m in LLMS:
        f = os.path.join(PROC, f"llm_cls_{m}.csv")
        if not os.path.exists(f):
            continue
        L = pd.read_csv(f)
        cols = [f"p_{c}" for c in CLASSES]
        for mode in ("zero", "few"):
            Lm = L[L["mode"] == mode].set_index("id")
            p = Lm.reindex(P.id)[cols]
            if p.isna().any().any():
                raise ValueError(f"missing LLM outputs for {m} {mode}")
            llm[(m, mode)] = p.values
            cf = Lm[Lm.index.str.startswith("CF_")][cols].values
            if len(cf):
                # contextual calibration: divide by the mean prediction for content-free inputs, renormalise
                w = 1.0 / np.clip(cf.mean(0), 1e-6, None)
                q = p.values * w
                llm[(m, mode + "+cc")] = q / q.sum(1, keepdims=True)
    return P, y, E, llm


def full_proba(model, X):
    """Predicted probabilities over all 12 classes (classes absent from the training split get probability 0)."""
    p = np.zeros((X.shape[0], K))
    p[:, model.classes_] = model.predict_proba(X)
    return p


def lr(C=10.0):
    return LogisticRegression(C=C, max_iter=1000, tol=1e-3)


def split_indices(P, design, rng):
    n = len(P)
    if design == "project":
        projs = P.project.unique()
        rng.shuffle(projs)
        sizes = P.project.value_counts()
        cum = np.cumsum([sizes[p] for p in projs]) / n
        tr_p, ca_p = set(projs[cum <= 0.5]), set(projs[(cum > 0.5) & (cum <= 0.75)])
        part = np.array([0 if p in tr_p else 1 if p in ca_p else 2 for p in P.project])
    else:
        part = np.repeat([0, 1, 2], [n // 2, n // 4, n - n // 2 - n // 4])
        rng.shuffle(part)
    return np.where(part == 0)[0], np.where(part == 1)[0], np.where(part == 2)[0]


def point_metrics(p, y):
    pred = p.argmax(1)
    b_true, b_pred = (y == 0), (pred == 0)              # class 0 = functional
    return {"acc": np.mean(pred == y), "macro_f1": f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0),
            "acc_fnfr": np.mean(b_true == b_pred), "f1_fnfr": f1_score(b_true, b_pred, average="macro", zero_division=0)}


def conformal_metrics(pc, yc, pt, yt):
    out = {}
    idx = np.arange(len(yt))
    for a in ALPHAS:
        s = lac_sets(pt, lac_threshold(pc, yc, a))
        out[f"cov_{a}"] = s[idx, yt].mean()
        out[f"size_{a}"] = s.sum(1).mean()
        out[f"single_{a}"] = (s.sum(1) == 1).mean()
    sm = mondrian_sets(pt, mondrian_thresholds(pc, yc, 0.1, K))
    out["mcov_0.1"], out["msize_0.1"] = sm[idx, yt].mean(), sm.sum(1).mean()
    sl = lac_sets(pt, lac_threshold(pc, yc, 0.1))
    per_class = {f"cls_cov_{CLASSES[c]}": (sl[idx, yt][yt == c].sum(), (yt == c).sum()) for c in range(K)}
    per_class_m = {f"cls_mcov_{CLASSES[c]}": (sm[idx, yt][yt == c].sum(), (yt == c).sum()) for c in range(K)}
    # risk-controlled automation with conformal risk control: share of all requirements auto-labelled and wrong <= b
    bc, bt = np.stack([pc[:, 0], 1 - pc[:, 0]], 1), np.stack([pt[:, 0], 1 - pt[:, 0]], 1)
    ybc, ybt = (yc != 0).astype(int), (yt != 0).astype(int)
    for b in BUDGET:
        for tag, (qc, qyc, qt, qyt) in (("12", (pc, yc, pt, yt)), ("2", (bc, ybc, bt, ybt))):
            lam = crc_automation_threshold(qc.max(1), qc.argmax(1) != qyc, b)
            acc = qt.max(1) >= lam
            wrong = acc & (qt.argmax(1) != qyt)
            out[f"crc{tag}_auto_{b}"] = acc.mean()
            out[f"crc{tag}_risk_{b}"] = wrong.mean()
            out[f"crc{tag}_selerr_{b}"] = wrong.sum() / acc.sum() if acc.any() else np.nan
    # risk-controlled automation (Learn-then-Test, high-probability guarantee on the selective error rate)
    for e in EPS12:
        lam = ltt_threshold(pc.max(1), pc.argmax(1) == yc, e, DELTA)
        acc = pt.max(1) >= lam
        out[f"auto12_{e}"] = acc.mean()
        out[f"err12_{e}"] = np.mean(pt.argmax(1)[acc] != yt[acc]) if acc.any() else np.nan
    for e in EPS2:
        lam = ltt_threshold(bc.max(1), bc.argmax(1) == ybc, e, DELTA)
        acc = bt.max(1) >= lam
        out[f"auto2_{e}"] = acc.mean()
        out[f"err2_{e}"] = np.mean(bt.argmax(1)[acc] != ybt[acc]) if acc.any() else np.nan
    return out, per_class, per_class_m


def run_split(P, y, E, llm, tr, ca, te, design):
    probs = {}
    # supervised baselines
    tf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1)
    Xtr = tf.fit_transform(P.text.values[tr])
    m = lr().fit(Xtr, y[tr])
    allx = tf.transform(P.text.values)
    probs["TF-IDF + LR"] = full_proba(m, allx)
    for name, key in (("MiniLM + LR", "minilm"), ("BGE + LR", "bge")):
        probs[name] = full_proba(lr().fit(E[key][tr], y[tr]), E[key])
    for (mname, mode), p in llm.items():
        probs[f"{LLM_LABEL[mname]} {MODE_LABEL[mode]}"] = p
    # fusion: out-of-fold BGE + LR probabilities on the training split, then a stacking model
    if llm:
        oof = np.zeros((len(tr), K))
        folds = GroupKFold(3).split(tr, groups=P.project.values[tr]) if design == "project" else \
            KFold(3, shuffle=True, random_state=SEED).split(tr)
        for a, b in folds:
            oof[b] = full_proba(lr().fit(E["bge"][tr[a]], y[tr[a]]), E["bge"][tr[b]])
        bge_all = probs["BGE + LR"]
        for mname in sorted({k[0] for k in llm}, key=list(LLMS).index):
            pz = llm[(mname, "zero")]
            Ftr = np.hstack([np.log(pz[tr] + EPS), np.log(oof + EPS)])
            Fall = np.hstack([np.log(pz + EPS), np.log(bge_all + EPS)])
            probs[f"{LLM_LABEL[mname]} + BGE fusion"] = full_proba(lr(C=1.0).fit(Ftr, y[tr]), Fall)
    # calibration-size experiment: how many labelled requirements are needed (subsamples of the calibration split)
    calsize = []
    rng = np.random.default_rng(len(tr) * 7919 + len(ca))
    order = rng.permutation(ca)
    for name, p in probs.items():
        if not (name in ("BGE + LR", "TF-IDF + LR") or "fusion" in name or name.endswith("zero-shot")):
            continue
        for ncal in CAL_SIZES:
            c = order[:ncal] if ncal < len(order) else order
            r = {"method": name, "n_cal": ncal if ncal < len(order) else -1, "n_cal_actual": len(c)}   # -1 = whole split
            q = lac_threshold(p[c], y[c], 0.1)
            st = lac_sets(p[te], q)
            r["cov_0.1"], r["size_0.1"] = st[np.arange(len(te)), y[te]].mean(), st.sum(1).mean()
            for b in (0.05, 0.1):
                lam = crc_automation_threshold(p[c].max(1), p[c].argmax(1) != y[c], b)
                acc = p[te].max(1) >= lam
                r[f"crc12_auto_{b}"], r[f"crc12_risk_{b}"] = acc.mean(), (acc & (p[te].argmax(1) != y[te])).mean()
            calsize.append(r)
    rows, pcs = [], []
    for name, p in probs.items():
        r = {"method": name}
        r.update(point_metrics(p[te], y[te]))
        cm, pc1, pc2 = conformal_metrics(p[ca], y[ca], p[te], y[te])
        r.update(cm)
        rows.append(r)
        pcs.append((name, pc1, pc2))
    return rows, pcs, calsize


def summarise(R):
    num = [c for c in R.columns if c not in ("method", "design", "split")]
    g = R.groupby(["design", "method"])[num]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    return out.reset_index()


def llm_full_sample(P, y, llm):
    """Accuracy of each generative model on all 957 evaluation requirements, with project-cluster bootstrap CIs."""
    rows, tests = [], []
    preds = {}
    for (m, mode), p in llm.items():
        pred = p.argmax(1)
        preds[(m, mode)] = pred == y
        a = cluster_bootstrap((pred == y).astype(float), P.project.values)
        b = cluster_bootstrap(((pred == 0) == (y == 0)).astype(float), P.project.values)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        conf = p.max(1)
        ece = np.sum([np.abs((pred == y)[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean()) *
                      ((conf > lo) & (conf <= hi)).mean() for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:])
                      if ((conf > lo) & (conf <= hi)).any()])
        rows.append({"model": LLM_LABEL[m], "prompt": MODE_LABEL[mode], "n": len(y),
                     "acc": a[0], "acc_lo": a[1], "acc_hi": a[2], "macro_f1": f1, "acc_fnfr": b[0], "acc_fnfr_lo": b[1],
                     "acc_fnfr_hi": b[2], "mean_conf": conf.mean(), "ece": ece})
    keys = list(preds)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            n10, n01, pv = mcnemar_exact(preds[keys[i]], preds[keys[j]])
            tests.append({"a": f"{LLM_LABEL[keys[i][0]]} {MODE_LABEL[keys[i][1]]}", "b": f"{LLM_LABEL[keys[j][0]]} {MODE_LABEL[keys[j][1]]}",
                          "acc_a": preds[keys[i]].mean(), "acc_b": preds[keys[j]].mean(), "a_only": n10, "b_only": n01, "p": pv})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def main():
    t0 = time.time()
    P, y, E, llm = load()
    print(f"{len(P)} requirements, {P.project.nunique()} projects, LLM outputs: {sorted(llm)}", flush=True)
    full, tests = llm_full_sample(P, y, llm)
    full.to_csv(os.path.join(TABLES, "cls_llm_full_sample.csv"), index=False)
    tests.to_csv(os.path.join(TABLES, "cls_llm_mcnemar.csv"), index=False)
    rows, cls_counts, calrows = [], {}, []
    for design in ("project", "random"):
        rng = np.random.default_rng(SEED)
        for s in range(N_SPLITS):
            tr, ca, te = split_indices(P, design, rng)
            r, pcs, cs = run_split(P, y, E, llm, tr, ca, te, design)
            for x in cs:
                x.update({"design": design, "split": s})
            calrows += cs
            for x in r:
                x.update({"design": design, "split": s})
            rows += r
            for name, pc1, pc2 in pcs:
                for k, (cov, n) in {**pc1, **pc2}.items():
                    acc = cls_counts.setdefault((design, name, k), [0, 0])
                    acc[0] += cov
                    acc[1] += n
            if s % 20 == 0:
                print(f"{design} split {s}/{N_SPLITS} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(PROC, "cls_split_results.csv"), index=False)
    CS = pd.DataFrame(calrows)
    CS.to_csv(os.path.join(PROC, "cls_calsize_results.csv"), index=False)
    g = CS.groupby(["design", "method", "n_cal"])
    g.mean(numeric_only=True).drop(columns="split").add_suffix("_mean").join(g.std(numeric_only=True).drop(columns="split").add_suffix("_sd")) \
        .reset_index().to_csv(os.path.join(TABLES, "cls_calibration_size.csv"), index=False)
    S = summarise(R)
    S.to_csv(os.path.join(TABLES, "cls_summary.csv"), index=False)
    C = pd.DataFrame([{"design": d, "method": m, "kind": "Mondrian" if "mcov" in k else "marginal",
                       "class": k.split("_")[-1], "covered": v[0], "n": v[1], "coverage": v[0] / v[1] if v[1] else np.nan}
                      for (d, m, k), v in cls_counts.items()])
    C.to_csv(os.path.join(TABLES, "cls_class_coverage.csv"), index=False)
    # paired comparison of each fusion with its own LLM and with BGE + LR (project design), sign test over splits
    comp = []
    Rp = R[R.design == "project"].set_index(["split", "method"])
    for m in LLMS:
        lab = LLM_LABEL[m]
        f = f"{lab} + BGE fusion"
        if (0, f) not in Rp.index:
            continue
        for other in (f"{lab} zero-shot", "BGE + LR"):
            for metric in ("acc", "macro_f1", "size_0.1", "crc12_auto_0.05", "crc12_auto_0.1", "crc2_auto_0.02", "crc2_auto_0.05"):
                a = Rp.xs(f, level="method")[metric]
                b = Rp.xs(other, level="method")[metric]
                d = (a - b).dropna()
                pv = stats.binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue if (d != 0).any() else 1.0
                comp.append({"fusion": f, "vs": other, "metric": metric, "mean_diff": d.mean(), "sd_diff": d.std(),
                             "share_better": (d > 0).mean(), "share_worse": (d < 0).mean(), "p_sign": pv})
    pd.DataFrame(comp).to_csv(os.path.join(TABLES, "cls_paired_comparisons.csv"), index=False)
    print(S[S.design == "project"][["method", "acc_mean", "macro_f1_mean", "cov_0.1_mean", "size_0.1_mean", "crc12_auto_0.05_mean",
                                    "crc12_risk_0.05_mean", "crc12_auto_0.1_mean", "crc2_auto_0.02_mean", "crc2_auto_0.05_mean",
                                    "auto12_0.2_mean"]].round(3).to_string())
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
