"""Step 07 - Figures 2-7 (300 dpi PNG). Colour encodes the method family (fixed order, colour-blind-checked palette);
markers and direct labels carry the same information for print."""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, FIGS, LLM_LABEL, LLMS, TABLES, TRACE_SETS  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILY = {"lightweight": ("#4a3aa7", "s"), "llm": ("#eb6834", "o"), "llm_cc": ("#eda100", "D"), "fusion": ("#2a78d6", "^"),
          "cross": ("#1baf7a", "v")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def family(m):
    if "fusion" in m or m.endswith("+ RRF"):
        return "fusion"
    if "calibration" in m:
        return "llm_cc"
    if any(LLM_LABEL[k] in m for k in LLMS):
        return "llm"
    if m == "Cross-encoder":
        return "cross"
    return "lightweight"


def style(m):
    """Colour and marker per method for line and point charts (identity, fixed order)."""
    if m.endswith("+ retrieval fusion") or m.endswith("+ BGE fusion") or m.endswith("+ RRF"):
        return "#2a78d6", "^"
    if m == "Retrieval fusion":
        return "#eda100", "D"
    if m in ("TF-IDF + LR", "Cross-encoder"):
        return "#1baf7a", "v"
    if m in ("BGE + LR", "RRF (BM25 + BGE)"):
        return "#4a3aa7", "s"
    return "#eb6834", "o"


def method_handles(methods, **kw):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=style(m)[0], marker=style(m)[1], markersize=6, label=m, **kw) for m in methods]


def legend_handles(fams, labels):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=FAMILY[f][0], marker=FAMILY[f][1], linestyle="none", markersize=6, label=l) for f, l in zip(fams, labels)]


def fig2_accuracy(S):
    """Dot plot: 12-class accuracy and macro-F1 (project design), mean +- SD over splits."""
    s = S[S.design == "project"].sort_values("acc_mean")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.24 * len(s) + 1.0), sharey=True)
    for ax, (col, lab) in zip(axes, [("acc", "Accuracy (12 classes)"), ("macro_f1", "Macro-F1 (12 classes)"), ("acc_fnfr", "Accuracy (F/NFR)")]):
        for i, r in enumerate(s.itertuples()):
            c, mk = FAMILY[family(r.method)]
            ax.errorbar(getattr(r, f"{col}_mean"), i, xerr=getattr(r, f"{col}_sd"), fmt=mk, color=c, ms=5, elinewidth=1, capsize=0)
        ax.set_xlabel(lab)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(s)))
    axes[0].set_yticklabels(s.method)
    fig.legend(handles=legend_handles(["lightweight", "llm", "llm_cc", "fusion"], ["Lightweight supervised", "LLM", "LLM + contextual calibration", "LLM + BGE fusion"]),
               loc="upper center", ncol=4, bbox_to_anchor=(0.55, 1.02))
    fig.savefig(os.path.join(FIGS, "Fig2_classification_accuracy.png"))
    plt.close(fig)


def fig3_coverage(S, pick):
    """Coverage and set size of conformal sets at alpha = 0.1 under both designs."""
    s = S[S.method.isin(pick)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.3 * len(pick) + 1.2), sharey=True)
    for j, (ax, col, lab) in enumerate(zip(axes, ["cov_0.1", "size_0.1"], ["Empirical coverage (target 0.90)", "Mean set size (of 12 classes)"])):
        for i, m in enumerate(pick):
            for k, (design, dy) in enumerate([("project", -0.15), ("random", 0.15)]):
                r = s[(s.method == m) & (s.design == design)]
                if r.empty:
                    continue
                c = FAMILY[family(m)][0]
                ax.errorbar(r[f"{col}_mean"].iloc[0], i + dy, xerr=r[f"{col}_sd"].iloc[0], fmt="o" if design == "project" else "s",
                            mfc=c if design == "project" else "white", color=c, ms=5, elinewidth=1)
        if col == "cov_0.1":
            ax.axvline(0.9, color=INK2, lw=1, ls="--")
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(pick)))
    axes[0].set_yticklabels(pick)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", color=INK2, mfc=INK2, ls="none", label="New projects (project design)"),
                        Line2D([0], [0], marker="s", color=INK2, mfc="white", ls="none", label="Random requirements")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.03))
    fig.savefig(os.path.join(FIGS, "Fig3_conformal_sets.png"))
    plt.close(fig)


def fig4_automation(S, pick):
    """Share of requirements labelled automatically at a guaranteed error budget (conformal risk control)."""
    s = S[S.design == "project"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    budgets = [0.02, 0.05, 0.1]
    for ax, task, title in zip(axes, ["12", "2"], ["12 classes", "Functional vs non-functional"]):
        for m in pick:
            r = s[s.method == m]
            if r.empty:
                continue
            c, mk = style(m)
            y = [r[f"crc{task}_auto_{b}_mean"].iloc[0] for b in budgets]
            e = [r[f"crc{task}_auto_{b}_sd"].iloc[0] for b in budgets]
            ax.errorbar([100 * b for b in budgets], y, yerr=e, marker=mk, color=c, lw=1.6, ms=5, capsize=2, elinewidth=0.8)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks([2, 5, 10])
        ax.set_xlim(1, 11)
    axes[0].set_ylabel("Share labelled automatically")
    axes[0].set_ylim(0, 1)
    fig.supxlabel("Error budget ε: guaranteed expected share of all requirements labelled automatically and wrongly (%)", fontsize=8.5, y=-0.03)
    fig.legend(handles=method_handles(pick), loc="upper center", ncol=len(pick), bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    fig.savefig(os.path.join(FIGS, "Fig4_automation.png"))
    plt.close(fig)


def fig5_class_coverage(C, method, counts):
    """Per-class coverage of marginal versus class-conditional (Mondrian) sets at alpha = 0.1 (project design)."""
    c = C[(C.design == "project") & (C.method == method)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(CLASSES))
    for k, (kind, col, dx) in enumerate([("marginal", "#eb6834", -0.18), ("Mondrian", "#2a78d6", 0.18)]):
        v = [c[(c.kind == kind) & (c["class"] == cl)].coverage.iloc[0] for cl in CLASSES]
        ax.bar(x + dx, v, width=0.34, color=col, label="Marginal sets" if kind == "marginal" else "Class-conditional (Mondrian) sets")
    ax.axhline(0.9, color=INK2, lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cl}\n(n={counts[cl]})" for cl in CLASSES], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage (α = 0.10)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    fig.savefig(os.path.join(FIGS, "Fig6_class_coverage.png"))
    plt.close(fig)


def fig6_trace_curves(curves, crc, pick):
    """Links to vet per requirement versus recall within the K = 20 shortlist, with the conformal operating points."""
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=False)
    for ax, ds in zip(axes, TRACE_SETS):
        for m in pick:
            cu = curves[(curves.universe == "shortlist") & (curves.dataset == ds) & (curves.method == m)].sort_values("recall")
            if cu.empty:
                continue
            c, mk = style(m)
            ax.plot(cu.recall, cu.kept_per_req, color=c, lw=1.4)
            op = crc[(crc.universe == "shortlist") & (crc.dataset == ds) & (crc.method == m) & (crc.alpha == 0.1)]
            if not op.empty:
                ax.plot(op.recall_mean, op.kept_per_req, marker=mk, color=c, ms=6, mec="white", mew=0.8)
        ax.axvline(0.9, color=INK2, lw=0.8, ls="--")
        ax.set_title(ds, fontsize=9)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Links to vet per requirement")
    fig.supxlabel("Mean recall of the gold links in the shortlist", fontsize=8.5, y=-0.04)
    fig.legend(handles=method_handles(pick), loc="upper center", ncol=len(pick), bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
    fig.savefig(os.path.join(FIGS, "Fig5_trace_effort.png"))
    plt.close(fig)


def fig7_trace_validity(crc, shift, pick):
    """Realised recall at target 0.90: within-system calibration (mean and 5th percentile over splits) and
    cross-system calibration."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(len(TRACE_SETS))
    w = 0.8 / len(pick)
    for k, m in enumerate(pick):
        c, mk = style(m)
        within = [crc[(crc.universe == "all") & (crc.dataset == ds) & (crc.method == m) & (crc.alpha == 0.1)].recall_mean.iloc[0]
                  for ds in TRACE_SETS]
        ax.plot(x - 0.4 + w * (k + 0.5), within, marker=mk, color=c, ls="none", ms=6)
        sh = shift[(shift.method == m) & (shift.alpha == 0.1)]
        if not sh.empty:
            cross = [sh[sh.dataset == ds].recall.iloc[0] for ds in TRACE_SETS]
            ax.plot(x - 0.4 + w * (k + 0.5), cross, marker=mk, color=c, mfc="white", ls="none", ms=6)
    ax.axhline(0.9, color=INK2, lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(TRACE_SETS)
    ax.set_ylabel("Realised mean recall (target 0.90)")
    from matplotlib.lines import Line2D
    h = method_handles(pick, linestyle="none")
    h += [Line2D([0], [0], marker="o", color=INK2, ls="none", label="filled: calibrated on the same system"),
          Line2D([0], [0], marker="o", color=INK2, mfc="white", ls="none", label="open: calibrated on the other systems")]
    ax.legend(handles=h, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIGS, "Fig7_trace_validity.png"))
    plt.close(fig)


def main():
    S = pd.read_csv(os.path.join(TABLES, "cls_summary.csv"))
    C = pd.read_csv(os.path.join(TABLES, "cls_class_coverage.csv"))
    sp = S[S.design == "project"]
    fusions = sp[sp.method.str.contains("fusion")].sort_values("acc_mean")
    best_fusion = fusions.method.iloc[-1]
    llm_zero = sp[sp.method.str.endswith("zero-shot")].sort_values("acc_mean").method.iloc[-1]
    fig2_accuracy(S)
    cov_pick = ["TF-IDF + LR", "BGE + LR"] + [f"{LLM_LABEL[m]} zero-shot" for m in LLMS if f"{LLM_LABEL[m]} zero-shot" in set(S.method)] + [best_fusion]
    fig3_coverage(S, cov_pick)
    fig4_automation(S, ["TF-IDF + LR", "BGE + LR", llm_zero, best_fusion])
    P = pd.read_csv(os.path.join(os.path.dirname(TABLES), "..", "data", "processed", "promise_exp.csv"))
    fig5_class_coverage(C, best_fusion, P[P.exemplar == 0].label.value_counts().to_dict())
    f = os.path.join(TABLES, "trace_crc.csv")
    if os.path.exists(f):
        crc = pd.read_csv(f)
        curves = pd.read_csv(os.path.join(TABLES, "trace_tradeoff_curves.csv"))
        shift = pd.read_csv(os.path.join(TABLES, "trace_crc_shift.csv"))
        methods = list(crc.method.unique())
        T = pd.read_csv(os.path.join(TABLES, "trace_ranking.csv"))
        mmap = T.groupby("method").MAP.mean()
        llms = [m for m in methods if m.endswith("re-rank")]
        fus = [m for m in methods if m.endswith("+ retrieval fusion") or m.endswith("+ RRF")]
        pick = ["RRF (BM25 + BGE)", "Cross-encoder", max(llms, key=lambda m: mmap[m]), max(fus, key=lambda m: mmap[m])]
        fig6_trace_curves(curves, crc, pick)
        fig7_trace_validity(crc, shift, [m for m in pick if "fusion" not in m])
    print("figures written")


if __name__ == "__main__":
    main()
