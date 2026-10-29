"""Step 09 - Fig. 1: the risk-controlled LLM assistance framework (Graphviz)."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGS  # noqa: E402

DOT = r'''digraph G {
  graph [rankdir=LR, fontname="DejaVu Sans", fontsize=11, nodesep=0.28, ranksep=0.42, pad=0.2, bgcolor="white", newrank=true];
  node  [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=10, color="#52514e", penwidth=0.8, margin="0.13,0.07"];
  edge  [color="#52514e", arrowsize=0.7, penwidth=0.9, fontname="DejaVu Sans", fontsize=8, fontcolor="#52514e"];

  subgraph cluster_in { label="RE artefacts"; fontsize=10; fontcolor="#52514e"; color="#e4e3df"; style="rounded";
    req  [label="Requirements\n(natural language)", fillcolor="#f4f3ef"];
    code [label="Code artefacts\n(classes, files)", fillcolor="#f4f3ef"];
  }
  subgraph cluster_as { label="1  Assistants (scores)"; fontsize=10; fontcolor="#52514e"; color="#e4e3df"; style="rounded";
    llm  [label="Open-weight LLMs on a CPU\n(0.5B–3B, 4-bit)\nanswer-option log-probabilities", fillcolor="#dbe9fa"];
    lw   [label="Lightweight models\nTF-IDF / embedding classifiers\nBM25, BGE, cross-encoder", fillcolor="#dbe9fa"];
    fus  [label="Fusion\nstacking classifier /\ncross-project link model", fillcolor="#dbe9fa"];
  }
  cal [label="2  Calibration\nsmall labelled sample\n(requirements as units)", fillcolor="#fde3d6"];
  subgraph cluster_cp { label="3  Risk control (distribution-free)"; fontsize=10; fontcolor="#52514e"; color="#e4e3df"; style="rounded";
    ps   [label="Conformal prediction sets\ncoverage ≥ 1 − α", fillcolor="#d5f0e5"];
    ltt  [label="Learn-then-Test automation\nerror ≤ ε with probability ≥ 1 − δ", fillcolor="#d5f0e5"];
    crc  [label="Conformal risk control\nexpected trace recall ≥ 1 − α", fillcolor="#d5f0e5"];
  }
  subgraph cluster_out { label="4  Triage for the analyst"; fontsize=10; fontcolor="#52514e"; color="#e4e3df"; style="rounded";
    auto [label="Labelled automatically", fillcolor="#ffffff"];
    rev  [label="Reviewed with a small\nset of candidate labels", fillcolor="#ffffff"];
    vet  [label="Short list of trace links\nto vet (rest pruned)", fillcolor="#ffffff"];
  }
  eff [label="Analyst effort at a\nguaranteed error / recall", shape=box, style="rounded,filled,bold", fillcolor="#ffffff"];

  req -> llm; req -> lw; code -> llm; code -> lw;
  llm -> fus; lw -> fus;
  llm -> cal; lw -> cal; fus -> cal;
  cal -> ps; cal -> ltt; cal -> crc;
  ltt -> auto; ps -> rev; crc -> vet;
  auto -> eff; rev -> eff; vet -> eff;
}
'''


def main():
    src = os.path.join(FIGS, "Fig1_framework.dot")
    with open(src, "w") as fh:
        fh.write(DOT)
    subprocess.run(["dot", "-Tpng", "-Gdpi=300", src, "-o", os.path.join(FIGS, "Fig1_framework.png")], check=True)
    print("ok Fig1")


if __name__ == "__main__":
    main()
