"""Step 02 - build the analysis tables from the raw data.

Requirements classification: PROMISE_exp (Lima et al., 2019), 969 requirements from 47 projects, 12 classes.
Trace link recovery: requirements/use cases -> code classes for eTour, iTrust, SMOS (English translation) and LibEST,
as distributed with the fine-grained traceability benchmark of Hey et al. (2021).

Outputs (data/processed):
  promise_exp.csv     id, project, text, label, exemplar (1 = used as the single few-shot example of its class)
  trace_artifacts.csv dataset, kind (req/code), aid, text (full), summary (compact text shown to the LLM)
  trace_links.csv     dataset, req, code (gold links)
"""
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, PROC, RAW, SEED  # noqa: E402

FGT = os.path.join(RAW, "traceability")


# ---------------------------------------------------------------- PROMISE_exp
def build_promise():
    p = os.path.join(RAW, "promise_exp", "PROMISE_exp.arff")
    txt = open(p, encoding="utf-8", errors="replace").read()
    rows = []
    for ln in txt.split("@DATA", 1)[1].strip().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("%"):
            continue
        m = re.match(r"^(\d+),'(.*)',([A-Z]{1,2})\s*$", ln)
        if not m:
            raise ValueError("cannot parse: " + ln)
        rows.append({"project": int(m.group(1)), "text": re.sub(r"\s+", " ", m.group(2)).strip(), "label": m.group(3)})
    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"R{i + 1:04d}" for i in range(len(df))])
    assert set(df.label) == set(CLASSES)
    # one few-shot exemplar per class, drawn at random and removed from every evaluation
    rng = random.Random(SEED)
    df["exemplar"] = 0
    for c in CLASSES:
        ids = sorted(df.index[df.label == c])
        df.loc[rng.choice(ids), "exemplar"] = 1
    df.to_csv(os.path.join(PROC, "promise_exp.csv"), index=False)
    print(f"PROMISE_exp: {len(df)} requirements, {df.project.nunique()} projects, {df.exemplar.sum()} exemplars")
    print(df.label.value_counts().to_dict())


# ---------------------------------------------------------------- traceability
def read(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def split_ident(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.replace("_", " ")


JAVA_KW = {"if", "for", "while", "switch", "catch", "return", "new", "else", "try", "synchronized", "throw", "super", "this"}


def strip_comments(src):
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', src)


def clean_doc(c):
    c = re.sub(r"^\s*\*+", " ", c, flags=re.M)
    c = re.split(r"\s@(?:param|return|throws|author|version|see|since|exception)\b", c)[0]
    c = re.sub(r"<[^>]+>", " ", c)
    c = re.sub(r"[-=*_]{3,}", " ", c)
    return norm_ws(c)


def summarise_java(name, src, limit=600):
    desc = ""
    for c in re.findall(r"/\*\*(.*?)\*/", src, flags=re.S):
        c = clean_doc(c)
        if len(c) > 15 and "copyright" not in c.lower():
            desc = c
            break
    code = strip_comments(src)
    decl = re.search(r"\b(class|interface|enum)\s+(\w+)([^{]*)\{", code)
    head = (f"{decl.group(1)} {decl.group(2)} {norm_ws(decl.group(3))}").strip() if decl else name
    cname = decl.group(2) if decl else ""
    methods = []
    pat1 = r"^\s*(?:(?:public|protected|private|static|final|abstract|synchronized|native)\s+)+[\w<>\[\],\.\s?]*?\b(\w+)\s*\("
    pat2 = r"^\s*[\w<>\[\]\.]+\s+(\w+)\s*\([^;{)]*\)\s*(?:throws\s+[\w\.,\s]+)?;"
    for pat in (pat1, pat2):
        for m in re.finditer(pat, code, flags=re.M):
            nm = m.group(1)
            if nm not in JAVA_KW and nm not in methods and nm != cname:
                methods.append(nm)
    fields = []
    for m in re.finditer(r"^\s*(?:(?:private|protected|public|static|final)\s+)+[\w<>\[\],\.]+\s+(\w+)\s*[;=]", code, flags=re.M):
        nm = m.group(1)
        if nm not in fields and not nm.isupper() and nm != "serialVersionUID":
            fields.append(nm)
    parts = [f"Code class {name}: {head}."]
    if desc:
        parts.append("Description: " + desc[:250].rstrip(".") + ".")
    if methods:
        parts.append("Methods: " + ", ".join(methods[:25]) + ".")
    if fields:
        parts.append("Fields: " + ", ".join(fields[:12]) + ".")
    s = " ".join(parts)
    return s if len(s) <= limit else s[: limit - 3].rsplit(" ", 1)[0] + "..."


def summarise_c(name, src, limit=600):
    desc = ""
    for c in re.findall(r"/\*(.*?)\*/", src, flags=re.S)[:3]:
        if "@file" in c or "@brief" in c:
            d = clean_doc(re.sub(r"@(file|brief)\b\s*\S*", " ", c))
            if len(d) > 20 and "copyright" not in d.lower():
                desc = d
                break
    code = strip_comments(src)
    funcs = []
    for m in re.finditer(r"^[A-Za-z_][\w\s\*]*?\b([a-z_][a-z0-9_]+)\s*\([^;{]*?\)\s*[{;]", code, flags=re.M):
        nm = m.group(1)
        if nm not in funcs and nm not in JAVA_KW and nm not in ("sizeof", "defined"):
            funcs.append(nm)
    parts = [f"Code file {name}."]
    if desc:
        parts.append("Description: " + desc[:200].rstrip(".") + ".")
    if funcs:
        parts.append(("Function declarations: " if name.endswith(".h") else "Functions: ") + ", ".join(funcs[:35]) + ".")
    s = " ".join(parts)
    return s if len(s) <= limit else s[: limit - 3].rsplit(" ", 1)[0] + "..."


def code_bag(src):
    """Text used for lexical retrieval: identifiers split into words plus comments."""
    src = re.sub(r"^\s*(import|package|#include).*$", " ", src, flags=re.M)
    return norm_ws(split_ident(re.sub(r"[^A-Za-z0-9_]", " ", src)))


TRACE = {
    # name: (folder, req dir, code dir, links file)
    "eTour": ("eTour", "req", "code", "etour_solution_links_english.txt"),
    "iTrust": ("iTrust", "req", "code", "itrust_solution_links.txt"),
    "SMOS": ("smos", "trans_req", "trans_code", "smos_solution_links_italian.txt"),   # English translation, original file names
    "LibEST": ("LibEST", "req", "code", "req_to_code_ground.txt"),
}


def build_trace():
    arts, links = [], []
    for ds, (folder, rdir, cdir, lfile) in TRACE.items():
        base = os.path.join(FGT, folder)
        reqs = sorted(os.listdir(os.path.join(base, rdir)))
        code_paths = {}
        for root, _, files in os.walk(os.path.join(base, cdir)):
            for f in files:
                code_paths[f] = os.path.join(root, f)
        lower = {k.lower(): k for k in code_paths}
        for r in reqs:
            t = norm_ws(read(os.path.join(base, rdir, r)))
            arts.append({"dataset": ds, "kind": "req", "aid": r, "text": t, "summary": t[:1200]})
        for c, p in sorted(code_paths.items()):
            src = read(p)
            summ = summarise_c(c, src) if c.endswith((".c", ".h")) else summarise_java(c, src)
            arts.append({"dataset": ds, "kind": "code", "aid": c, "text": code_bag(src), "summary": summ})
        n_bad = 0
        for ln in read(os.path.join(base, lfile)).splitlines():
            if ":" not in ln:
                continue
            r, rest = ln.split(":", 1)
            for c in rest.split():
                c = c.strip()
                key = c if c in code_paths else lower.get(c.lower())
                if r.strip() in reqs and key:
                    links.append({"dataset": ds, "req": r.strip(), "code": key})
                else:
                    n_bad += 1
        nl = sum(1 for x in links if x["dataset"] == ds)
        print(f"{ds}: {len(reqs)} requirements, {len(code_paths)} code artifacts, {nl} links ({n_bad} unmatched entries dropped)")
    A = pd.DataFrame(arts)
    L = pd.DataFrame(links).drop_duplicates()
    A.to_csv(os.path.join(PROC, "trace_artifacts.csv"), index=False)
    L.to_csv(os.path.join(PROC, "trace_links.csv"), index=False)


if __name__ == "__main__":
    build_promise()
    build_trace()
