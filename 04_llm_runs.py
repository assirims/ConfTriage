"""Step 04 - query the generative models (llama.cpp, CPU, greedy decoding, fixed seed).

usage: python 04_llm_runs.py <model> <task>
  model: qwen0.5b | qwen1.5b | qwen3b | llama3b      (see config.LLMS)
  task:  cls   - requirements classification, zero-shot and few-shot (12 options, letters A-L)
         trace - trace-link verification of the top-K candidate code artifacts of every requirement (Yes/No)

The model answers with one token; we read the log-probabilities of the answer options at that position, so every
answer is a full probability distribution rather than a single label. Each prompt and response is appended to
results/llm/<task>_<model>.jsonl; an interrupted run resumes from that log without re-querying the model.
Outputs: data/processed/llm_cls_<model>.csv, data/processed/llm_trace_<model>.csv
"""
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, LLM_LOGS, LLMS, LOGS, PROC, SEED  # noqa: E402
from lib import llm  # noqa: E402

SYSTEM = "You are an experienced requirements engineer."
LETTERS = [chr(65 + i) for i in range(len(CLASSES))]
DEFS = {
    "F": "Functional: a behaviour or function that the system must perform for its users or other systems.",
    "A": "Availability: when the system must be available and how much downtime is acceptable.",
    "FT": "Fault tolerance: how the system must keep working, or recover, when failures or errors occur.",
    "L": "Legal: compliance with laws, regulations, standards, licences or contracts.",
    "LF": "Look and feel: the appearance, style and visual design of the user interface.",
    "MN": "Maintainability: how easily the system can be corrected, changed, updated or extended.",
    "O": "Operational: the operating environment, platforms, hardware or other systems the system must run in or work with.",
    "PE": "Performance: speed, response time, throughput, capacity or use of resources.",
    "PO": "Portability: the ability to run on, or be moved to, different platforms or environments.",
    "SC": "Scalability: the ability to handle growth in users, data or load without loss of service.",
    "SE": "Security: protection against unauthorised access, use or change; authentication, privacy and auditing.",
    "US": "Usability: how easy the system is to learn and use, and how efficient and satisfying it is for users.",
}
OPTIONS = "\n".join(f"{L}. {DEFS[c]}" for L, c in zip(LETTERS, CLASSES))


def cls_messages(text, examples=None):
    """Zero-shot: one user turn with the category definitions. Few-shot: the 12 exemplars are given as earlier
    question-answer turns of the same conversation (same wording), followed by the requirement to classify."""
    head = "Classify the software requirement below into exactly one of these categories:\n" + OPTIONS + "\n\n"
    ask = lambda t: f'Requirement: "{t}"\n\nAnswer with the letter of the category only.'
    msgs = [{"role": "system", "content": SYSTEM}]
    if examples:
        for k, (t, c) in enumerate(examples):
            msgs.append({"role": "user", "content": (head if k == 0 else "") + ask(t)})
            msgs.append({"role": "assistant", "content": LETTERS[CLASSES.index(c)]})
        msgs.append({"role": "user", "content": ask(text)})
    else:
        msgs.append({"role": "user", "content": head + ask(text)})
    return msgs


def few_shot_examples(P):
    """The 12 exemplars (one per class) in a fixed random order, so that the answers do not follow the option order."""
    ex = P[P.exemplar == 1].sample(frac=1.0, random_state=SEED)
    return [(r.text, r.label) for r in ex.itertuples()]


def trace_messages(req, code):
    """The instruction comes first, so that the shared prefix (instruction + requirement) is cached by the server and
    only the code artifact has to be processed for each candidate."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Decide whether the code artifact below is part of the implementation of the requirement, "
             "so that a trace link should connect them. Answer Yes or No.\n\nRequirement:\n" + req +
             "\n\nCode artifact:\n" + code + "\n\nAnswer (Yes or No):"}]


def option_logprobs(top, options):
    """Log-probabilities of the answer options at the first output position; options missing from the top list get
    a floor value one nat below the smallest listed log-probability."""
    first = top[0] if top else []
    lp = {}
    for tok, v in first:
        t = (tok or "").strip().strip(".:)").upper()
        if t in options and (t not in lp or v > lp[t]):
            lp[t] = v
    floor = (min(v for _, v in first) - 1.0) if first else -30.0
    return np.array([lp.get(o, floor) for o in options]), sum(o in lp for o in options)


def softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()


def run_cls(model):
    P = pd.read_csv(os.path.join(PROC, "promise_exp.csv"))
    ex = P[P.exemplar == 1]
    examples = few_shot_examples(P)
    log = os.path.join(LLM_LOGS, f"cls_{model}.jsonl")
    llm.load_replay(log)
    rows, t0 = [], time.time()
    for mode in ("zero", "few"):
        for i, r in enumerate(P.itertuples()):
            if mode == "few" and r.exemplar == 1:
                continue
            msgs = cls_messages(r.text, examples if mode == "few" else None)
            out = llm.chat(msgs, max_tokens=1, top_logprobs=20, log_file=log, tag=f"cls:{mode}:{r.id}")
            lp, found = option_logprobs(out["top_logprobs"], LETTERS)
            p = softmax(lp)
            row = {"id": r.id, "mode": mode, "pred": CLASSES[int(np.argmax(p))], "answer": out["text"], "n_options_found": found,
                   "seconds": out["seconds"]}
            row.update({f"p_{c}": p[k] for k, c in enumerate(CLASSES)})
            rows.append(row)
            if i % 100 == 0:
                print(f"{model} cls {mode} {i}/{len(P)} ({time.time() - t0:.0f}s)", flush=True)
        # content-free inputs for contextual calibration (Zhao et al., 2021)
        for cid, cf in (("CF_NA", "N/A"), ("CF_EMPTY", ""), ("CF_MASK", "[MASK]")):
            out = llm.chat(cls_messages(cf, examples if mode == "few" else None), max_tokens=1, top_logprobs=20, log_file=log,
                           tag=f"cls:{mode}:{cid}")
            lp, found = option_logprobs(out["top_logprobs"], LETTERS)
            p = softmax(lp)
            row = {"id": cid, "mode": mode, "pred": CLASSES[int(np.argmax(p))], "answer": out["text"], "n_options_found": found,
                   "seconds": out["seconds"]}
            row.update({f"p_{c}": p[k] for k, c in enumerate(CLASSES)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(PROC, f"llm_cls_{model}.csv"), index=False)


def run_trace(model):
    A = pd.read_csv(os.path.join(PROC, "trace_artifacts.csv"))
    S = {(d, k, a): s for d, k, a, s in zip(A.dataset, A.kind, A.aid, A.summary)}
    D = pd.read_csv(os.path.join(PROC, "trace_pairs.csv"))
    D = D[D.cand == 1].sort_values(["dataset", "req", "rrf_rank"])
    log = os.path.join(LLM_LOGS, f"trace_{model}.jsonl")
    llm.load_replay(log)
    rows, t0 = [], time.time()
    for i, r in enumerate(D.itertuples()):
        msgs = trace_messages(S[(r.dataset, "req", r.req)], S[(r.dataset, "code", r.code)])
        out = llm.chat(msgs, max_tokens=1, top_logprobs=20, log_file=log, tag=f"trace:{r.dataset}:{r.req}:{r.code}")
        lp, found = option_logprobs(out["top_logprobs"], ["YES", "NO"])
        p_yes = 1.0 / (1.0 + math.exp(lp[1] - lp[0]))
        rows.append({"dataset": r.dataset, "req": r.req, "code": r.code, "p_yes": p_yes, "logit_yes": lp[0] - lp[1],
                     "answer": out["text"], "n_options_found": found, "seconds": out["seconds"]})
        if i % 200 == 0:
            print(f"{model} trace {i}/{len(D)} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(PROC, f"llm_trace_{model}.csv"), index=False)


if __name__ == "__main__":
    model, task = sys.argv[1], sys.argv[2]
    proc = llm.start_server(ctx=4096, model=LLMS[model], log_path=os.path.join(LOGS, f"llama_server_{model}.log"))
    try:
        info = llm.model_info()
        print("model:", info, flush=True)
        if info.get("model_path") != os.path.basename(LLMS[model]):
            raise RuntimeError(f"a server with another model is running: {info}")
        run_cls(model) if task == "cls" else run_trace(model)
    finally:
        llm.stop_server(proc)
