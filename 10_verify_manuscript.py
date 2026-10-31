"""Step 10 - check the Word manuscript against the result files.

Every cell of Tables 2-17 is recomputed from results/tables/*.csv and data/processed/*.csv with the rounding used in
the paper (half up) and compared with the cell text; the key numbers quoted in the text are searched for; and every
listed reference must be cited.  Usage: python 10_verify_manuscript.py [path/to/manuscript.docx]
(default: manuscript/RELLM_Manuscript.docx).  Exit status 1 if anything does not match.
"""
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import docx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, LLM_LABEL, LLMS, PROC, RESULTS, TABLES, TRACE_SETS  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- reading the manuscript
def read_manuscript(path):
    """Paragraph text, table text and the tables as lists of rows (merged cells appear once)."""
    d = docx.Document(path)
    paras = "\n".join(p.text for p in d.paragraphs)
    tables = []
    for t in d.tables:
        rows = []
        for r in t.rows:
            cells, last = [], None
            for c in r.cells:
                if c._tc is not last:
                    cells.append(" ".join(p.text for p in c.paragraphs).strip())
                last = c._tc
            rows.append(cells)
        tables.append(rows)
    table_text = "\n".join(" | ".join(r) for t in tables for r in t)
    return paras, table_text, tables


# ----------------------------------------------------------------------------- formatting as in the paper
def fx(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def pct(x, d=0):
    return fx(100 * float(x), d) + "%"


def sign(d, s):
    return ("+" if d >= 0 else "-") + s


def norm(s):
    return str(s).replace("−", "-").replace("–", "-").replace(" ", " ").replace(" ", " ").replace(",", "").strip()


class Checker:
    def __init__(self):
        self.n, self.bad = 0, []

    def cmp(self, table, row, col, expected, got):
        self.n += 1
        if norm(expected) != norm(got):
            self.bad.append((table, row, col, expected, got))


# ----------------------------------------------------------------------------- table checks
def check_tables(tabs, ck):
    T = {i + 1: t for i, t in enumerate(tabs)}         # Table 1 is the literature table (not checked)
    DS = TRACE_SETS
    # Table 2: data sets
    D = pd.read_csv(os.path.join(PROC, "trace_pairs.csv"))
    for r in T[2][1:]:
        ds = r[0].strip()
        g = D[D.dataset == ds]
        per = g.groupby("req").gold.sum()
        ck.cmp(2, ds, "requirements", r[3], g.req.nunique())
        ck.cmp(2, ds, "targets", r[4], g.code.nunique())
        ck.cmp(2, ds, "links", r[5], int(g.gold.sum()))
        ck.cmp(2, ds, "links/req", r[6], fx(per[per > 0].mean(), 1))
        ck.cmp(2, ds, "shortlist recall", r[7], pct(g[g.cand == 1].gold.sum() / g.gold.sum()))
    # Table 3: LLMs on all evaluation requirements
    F = pd.read_csv(os.path.join(TABLES, "cls_llm_full_sample.csv"))
    for r in T[3][1:]:
        model, prompt = r[0].strip(), r[1].strip()
        x = F[(F.model == model) & (F.prompt == prompt)]
        if x.empty:
            ck.bad.append((3, f"{model} | {prompt}", "row", "present", "missing"))
            continue
        x = x.iloc[0]
        k = f"{model} | {prompt}"
        ck.cmp(3, k, "accuracy", r[2], f"{fx(x.acc)} ({fx(x.acc_lo)}–{fx(x.acc_hi)})")
        ck.cmp(3, k, "macro-F1", r[3], fx(x.macro_f1))
        ck.cmp(3, k, "F/NFR", r[4], f"{fx(x.acc_fnfr)} ({fx(x.acc_fnfr_lo)}–{fx(x.acc_fnfr_hi)})")
        ck.cmp(3, k, "confidence", r[5], fx(x.mean_conf))
        ck.cmp(3, k, "ECE", r[6], fx(x.ece))
    # Tables 4, 9, 11: split summaries
    S = pd.read_csv(os.path.join(TABLES, "cls_summary.csv"))
    Sp = S[S.design == "project"].set_index("method")
    Sr = S[S.design == "random"].set_index("method")

    def msd(df, m, c):
        return f"{fx(df.loc[m, c + '_mean'])} ({fx(df.loc[m, c + '_sd'])})"
    for r in T[4][2:]:
        m = r[0].strip()
        for j, (df, c) in enumerate([(Sp, "acc"), (Sp, "macro_f1"), (Sp, "acc_fnfr"), (Sr, "acc"), (Sr, "macro_f1"), (Sr, "acc_fnfr")]):
            ck.cmp(4, m, c, r[1 + j], msd(df, m, c))
    for r in T[9][2:]:
        m = r[0].strip()
        ck.cmp(9, m, "coverage", r[1], msd(Sp, m, "cov_0.1"))
        ck.cmp(9, m, "Mondrian coverage", r[2], fx(Sp.loc[m, "mcov_0.1_mean"]))
        ck.cmp(9, m, "risk 12", r[3], pct(Sp.loc[m, "crc12_risk_0.05_mean"], 1))
        ck.cmp(9, m, "risk F/NFR", r[4], pct(Sp.loc[m, "crc2_risk_0.05_mean"], 1))
        e = Sp.loc[m, "err12_0.2_mean"]
        ck.cmp(9, m, "LTT error", r[5], "–" if np.isnan(e) else pct(e, 1))
        ck.cmp(9, m, "random coverage", r[6], msd(Sr, m, "cov_0.1"))
        ck.cmp(9, m, "random risk", r[7], pct(Sr.loc[m, "crc12_risk_0.05_mean"], 1))
    for r in T[11][2:]:
        m = r[0].strip()
        ck.cmp(11, m, "set size", r[1], fx(Sp.loc[m, "size_0.1_mean"], 2))
        ck.cmp(11, m, "singletons", r[2], pct(Sp.loc[m, "single_0.1_mean"], 1))
        for j, c in enumerate(["crc12_auto_0.02", "crc12_auto_0.05", "crc12_auto_0.1", "crc2_auto_0.02", "crc2_auto_0.05", "auto12_0.2"]):
            ck.cmp(11, m, c, r[3 + j], pct(Sp.loc[m, c + "_mean"], 1))
        ck.cmp(11, m, "random set size", r[9], fx(Sr.loc[m, "size_0.1_mean"], 2))
        ck.cmp(11, m, "random automated", r[10], pct(Sr.loc[m, "crc12_auto_0.05_mean"], 1))
    # Table 5: per-class F1
    PC = pd.read_csv(os.path.join(TABLES, "cls_per_class.csv"))
    for r in T[5][1:]:
        m = r[0].strip()
        if m == "n":
            for j, c in enumerate(CLASSES):
                ck.cmp(5, "n", c, r[1 + j], int(PC[PC["class"] == c].n.iloc[0]))
            continue
        x = PC[PC.method == m].set_index("class")
        for j, c in enumerate(CLASSES):
            ck.cmp(5, m, c, r[1 + j], fx(x.loc[c, "f1"], 2))
    # Table 6: fusion comparisons (Holm-adjusted sign tests)
    P6 = pd.read_csv(os.path.join(TABLES, "cls_paired_comparisons.csv"))
    pcol = "p_holm" if "p_holm" in P6.columns else "p_sign"
    mets = ["acc", "macro_f1", "size_0.1", "crc12_auto_0.05", "crc12_auto_0.1", "crc2_auto_0.02", "crc2_auto_0.05"]
    for r in T[6][1:]:
        lab, vs = r[0].strip(), r[1].strip()
        fus = f"{lab} + BGE fusion"
        other = f"{lab} zero-shot" if vs == "LLM alone" else vs
        for j, met in enumerate(mets):
            x = P6[(P6.fusion == fus) & (P6.vs == other) & (P6.metric == met)].iloc[0]
            d = x.mean_diff
            s = sign(d, fx(abs(d), 2)) if met == "size_0.1" else sign(d, fx(abs(100 * d), 1))
            s += f" ({fx(100 * x.share_better, 0)}/{fx(100 * x.share_worse, 0)})" + ("†" if x[pcol] < 0.05 else "")
            ck.cmp(6, f"{lab} vs {vs}", met, r[2 + j], s)
    # Tables 7, 8: ranking quality
    TR = pd.read_csv(os.path.join(TABLES, "trace_ranking.csv")).set_index(["dataset", "method"])
    TS = pd.read_csv(os.path.join(TABLES, "trace_ranking_shortlist.csv")).set_index(["dataset", "method"])
    for r in T[7][1:]:
        m = r[0].strip()
        for j, ds in enumerate(DS):
            x = TR.loc[(ds, m)]
            ck.cmp(7, m, ds, r[1 + j], f"{fx(x.MAP)} ({fx(x.MAP_lo)}–{fx(x.MAP_hi)})")
        ck.cmp(7, m, "mean", r[5], fx(np.mean([TR.loc[(ds, m)].MAP for ds in DS])))
    for r in T[8][2:]:
        m = r[0].strip()
        for j, ds in enumerate(DS):
            x = TR.loc[(ds, m)]
            ck.cmp(8, m, f"{ds} R@5/R@10", r[1 + j], f"{fx(x.r5)} / {fx(x.r10)}")
            ck.cmp(8, m, f"{ds} shortlist MAP", r[5 + j], fx(TS.loc[(ds, m)].MAP))
    # Table 10: recall control within a system; Table 16: calibration on 10/20/30 requirements
    C = pd.read_csv(os.path.join(TABLES, "trace_crc.csv")).set_index(["universe", "dataset", "method", "alpha"])
    uni = alpha = None
    for r in T[10][1:]:
        if len(r) == 1:
            uni = "shortlist" if r[0].startswith("Shortlist") else "all"
            alpha = 0.1 if "0.90" in r[0] else 0.2
            continue
        m = r[0].strip()
        for j, ds in enumerate(DS):
            x = C.loc[(uni, ds, m, alpha)]
            ck.cmp(10, f"{uni} / {alpha} / {m}", ds, r[1 + j], f"{fx(x.kept_per_req, 1)} ({fx(x.recall_mean)})")
    CS = pd.read_csv(os.path.join(TABLES, "trace_crc_calsize.csv")).set_index(["universe", "dataset", "method", "alpha", "n_cal"])
    uni = None
    for r in T[16][2:]:
        if len(r) == 1:
            uni = "shortlist" if r[0].startswith("Shortlist") else "all"
            continue
        lab = r[0].strip()
        n = -1 if lab.startswith("half") else int(lab.split()[0])
        for j, ds in enumerate(DS):
            for k, m in enumerate(["RRF (BM25 + BGE)", "Qwen2.5-3B + RRF"]):
                x = CS.loc[(uni, ds, m, 0.1, n)]
                ck.cmp(16, f"{uni} / {lab} / {m}", ds, r[1 + j + 4 * k], f"{fx(x.recall_mean)} ({fx(x.kept_per_req, 1)})")
    # Table 12: calibration sample size (classification)
    CZ = pd.read_csv(os.path.join(TABLES, "cls_calibration_size.csv"))
    CZ = CZ[CZ.design == "project"].set_index(["method", "n_cal"])
    for r in T[12][2:]:
        m = r[0].strip()
        for j, nc in enumerate([30, 60, 120, -1]):
            x = CZ.loc[(m, nc)]
            ck.cmp(12, m, f"coverage/size {nc}", r[1 + j], f"{fx(x['cov_0.1_mean'])} / {fx(x['size_0.1_mean'], 2)}")
            ck.cmp(12, m, f"automated/risk {nc}", r[5 + j], f"{fx(100 * x['crc12_auto_0.05_mean'], 1)} / {fx(100 * x['crc12_risk_0.05_mean'], 1)}")
    # Table 13: class-conditional coverage
    CC = pd.read_csv(os.path.join(TABLES, "cls_class_coverage.csv"))
    CC = CC[CC.design == "project"].set_index(["method", "kind", "class"])
    for r in T[13][1:]:
        m, kind = r[0].strip(), r[1].strip()
        for j, c in enumerate(CLASSES):
            ck.cmp(13, f"{m} / {kind}", c, r[2 + j], fx(CC.loc[(m, kind, c)].coverage, 2))
    # Table 14: marginal versus class-conditional risk control
    CL = pd.read_csv(os.path.join(TABLES, "cls_class_conditional.csv"))
    CL = CL[CL.design == "project"].set_index("method")
    cols = ["marginal_auto", "marginal_risk", "marginal_worst_class", "cc_auto", "cc_risk", "cc_worst_class",
            "fnfr_marginal_auto", "fnfr_marginal_worst_class", "fnfr_cc_auto", "fnfr_cc_worst_class"]
    for r in T[14][2:]:
        m = r[0].strip()
        for j, c in enumerate(cols):
            ck.cmp(14, m, c, r[1 + j], pct(CL.loc[m, c], 1))
    # Table 15: thresholds transferred from the other systems († = target missed)
    SH = pd.read_csv(os.path.join(TABLES, "trace_crc_shift.csv")).set_index(["method", "dataset", "alpha"])
    for r in T[15][2:]:
        m = r[0].strip()
        for k, a in enumerate([0.1, 0.2]):
            for j, ds in enumerate(DS):
                x = SH.loc[(m, ds, a)]
                ck.cmp(15, f"{m} / {a}", ds, r[1 + j + 4 * k], f"{fx(x.recall)}{'' if x.met else '†'} ({fx(x.kept_per_req, 0)})")
    # Table 17: timing
    for r in T[17][1:]:
        model, task = r[0].strip(), r[1].strip()
        key = next(k for k, v in LLM_LABEL.items() if v == model)
        f = os.path.join(PROC, f"llm_{'cls' if task == 'Classification' else 'trace'}_{key}.csv")
        if not os.path.exists(f):
            continue
        x = pd.read_csv(f)
        ck.cmp(17, f"{model} / {task}", "queries", r[2], f"{len(x):,}")
        ck.cmp(17, f"{model} / {task}", "mean", r[3], fx(x.seconds.mean(), 2))
        ck.cmp(17, f"{model} / {task}", "median", r[4], fx(x.seconds.median(), 2))


# ----------------------------------------------------------------------------- numbers quoted in the text
def key_numbers(N):
    d = N["data"]
    checks = [("requirements", f"{d['cls_n']:,}"), ("projects", str(d["cls_projects"])), ("links", f"{d['trace_links']:,}"),
              ("trace requirements", str(d["trace_reqs"]))]
    S = [r for r in N["cls_summary"] if r["design"] == "project"]
    best = max((r for r in S if r["method"].endswith("BGE fusion")), key=lambda r: r["acc_mean"])
    bge = next(r for r in S if r["method"] == "BGE + LR")
    for r, lab in ((best, "best fusion"), (bge, "BGE + LR")):
        checks += [(f"{lab} accuracy", fx(r["acc_mean"])), (f"{lab} macro-F1", fx(r["macro_f1_mean"])),
                   (f"{lab} automated 5%", pct(r["crc12_auto_0.05_mean"])), (f"{lab} F/NFR automated 5%", pct(r["crc2_auto_0.05_mean"]))]
    cc = [r for r in N.get("cls_class_conditional", []) if r["design"] == "project" and r["method"] == best["method"]]
    if cc:
        checks += [("best fusion class-conditional automated", pct(cc[0]["cc_auto"])),
                   ("best fusion class-conditional worst class", pct(cc[0]["cc_worst_class"], 1)),
                   ("best fusion marginal worst class", pct(cc[0]["marginal_worst_class"]))]
    for r in N["cls_full"]:
        if r["prompt"] == "zero-shot":
            checks.append((f"{r['model']} zero-shot accuracy", fx(r["acc"])))
    if N.get("trace_ranking"):
        T = pd.DataFrame(N["trace_ranking"])
        for m in ("RRF (BM25 + BGE)", "Retrieval fusion", "Qwen2.5-3B + RRF"):
            checks.append((f"mean MAP {m}", fx(T[T.method == m].MAP.mean())))
        sh = pd.DataFrame(N["trace_shift"])
        for a in (0.1, 0.2):
            x = sh[sh.alpha == a]
            checks.append((f"transferred thresholds missed (alpha {a})", f"{int((~x.met.astype(bool)).sum())} of {len(x)}"))
        crc = pd.DataFrame(N["trace_crc"])
        checks.append(("share of splits that met the target (min)", pct(crc.share_splits_met.min())))
        checks.append(("share of splits that met the target (max)", pct(crc.share_splits_met.max())))
        cs = pd.DataFrame(N.get("trace_calsize", []))
        if len(cs):
            x = cs[(cs.n_cal == 20) & (cs.alpha == 0.1)]
            checks += [("recall with 20 requirements (min)", fx(x.recall_mean.min())), ("recall with 20 requirements (max)", fx(x.recall_mean.max())),
                       ("splits met with 20 requirements (min)", pct(x.share_splits_met.min())), ("splits met with 20 requirements (max)", pct(x.share_splits_met.max()))]
    return checks


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manuscript", "RELLM_Manuscript.docx")
    paras, table_text, tabs = read_manuscript(path)
    txt = paras + "\n" + table_text
    ck = Checker()
    check_tables(tabs, ck)
    print(f"{ck.n} table cells checked, {len(ck.bad)} mismatches")
    for b in ck.bad:
        print("  MISMATCH table", *b)
    N = json.load(open(os.path.join(RESULTS, "paper_numbers.json")))
    checks = key_numbers(N)
    ok = 0
    for lab, s in checks:
        found = s in txt
        ok += found
        if not found:
            print("NOT FOUND:", lab, s)
    body, _, refs = paras.partition("\nReferences\n")
    refs = refs.split("\nAppendix A")[0]
    body = body + "\n" + table_text
    cited = set()
    for m in re.finditer(r"\[(\d[\d,–]*)\]", body):
        for part in m.group(1).split(","):
            if "–" in part:
                a, b = map(int, part.split("–"))
                cited.update(range(a, b + 1))
            else:
                cited.add(int(part))
    n_refs = len(re.findall(r"^\[\d+\]\t", refs, flags=re.M))
    print(f"{ok}/{len(checks)} key numbers found in {os.path.relpath(path, ROOT)}; references listed: {n_refs}; distinct numbers cited: {len(cited)}")
    missing = sorted(set(range(1, n_refs + 1)) - cited)
    if missing:
        print("listed but not cited:", missing)
    if ck.bad or ok < len(checks) or missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
