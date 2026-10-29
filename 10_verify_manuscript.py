"""Step 10 - check that key numbers in the Word manuscript match the result tables, and that every reference is cited."""
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import docx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def text_of(path):
    """Return (paragraph text, table text); citations inside tables count as citations."""
    d = docx.Document(path)
    paras = "\n".join(p.text for p in d.paragraphs)
    tables = "\n".join(" | ".join(c.text for c in r.cells) for t in d.tables for r in t.rows)
    return paras, tables


def fx(x, d=3):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def pct(x, d=0):
    return fx(100 * float(x), d) + "%"


def main():
    N = json.load(open(os.path.join(RESULTS, "paper_numbers.json")))
    path = os.path.join(ROOT, "00_PAPER", "RELLM_Manuscript.docx")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "manuscript", "RELLM_Manuscript.docx")
    paras, tables = text_of(path)
    txt = paras + "\n" + tables
    d = N["data"]
    checks = [("requirements", f"{d['cls_n']:,}"), ("projects", str(d["cls_projects"])), ("links", f"{d['trace_links']:,}"),
              ("trace requirements", str(d["trace_reqs"]))]
    S = [r for r in N["cls_summary"] if r["design"] == "project"]
    best = max((r for r in S if r["method"].endswith("BGE fusion")), key=lambda r: r["acc_mean"])
    bge = next(r for r in S if r["method"] == "BGE + LR")
    for r, lab in ((best, "best fusion"), (bge, "BGE + LR")):
        checks += [(f"{lab} accuracy", fx(r["acc_mean"])), (f"{lab} macro-F1", fx(r["macro_f1_mean"])),
                   (f"{lab} automated 5%", pct(r["crc12_auto_0.05_mean"])), (f"{lab} F/NFR automated 5%", pct(r["crc2_auto_0.05_mean"]))]
    for r in N["cls_full"]:
        if r["prompt"] == "zero-shot":
            checks.append((f"{r['model']} zero-shot accuracy", fx(r["acc"])))
    if N.get("trace_ranking"):
        T = N["trace_ranking"]
        for r in T:
            if r["method"] in ("RRF (BM25 + BGE)", "Retrieval fusion"):
                checks.append((f"{r['dataset']} {r['method']} MAP", fx(r["MAP"])))
    ok = 0
    for lab, s in checks:
        found = s in txt
        ok += found
        if not found:
            print("NOT FOUND:", lab, s)
    body, _, refs = paras.partition("\nReferences\n")
    refs = refs.split("\nAppendix A")[0]
    body = body + "\n" + tables
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
    if ok < len(checks) or cited != set(range(1, n_refs + 1)):
        missing = sorted(set(range(1, n_refs + 1)) - cited)
        if missing:
            print("listed but not cited:", missing)
        sys.exit(1)


if __name__ == "__main__":
    main()
