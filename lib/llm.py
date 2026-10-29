"""Thin client for a local llama.cpp server (llama-server) running an open-weight instruction-tuned model (GGUF, Q4_K_M).

All generation is deterministic: temperature 0 (greedy), fixed seed, prompt caching on.
Every request and response is appended to a JSONL log so that each model output in the
paper can be traced back to its exact prompt.
"""
import json
import os
import subprocess
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/home/claude/tools/llama.cpp/build/bin/llama-server")
MODEL = os.environ.get("LLM_GGUF", os.path.join(ROOT, "models", "qwen2.5-1.5b-instruct-gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf"))
PORT = int(os.environ.get("LLM_PORT", "8089"))
URL = f"http://127.0.0.1:{PORT}"


def _post(path, payload, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def healthy():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(ctx=4096, threads=2, log_path=None, model=None):
    if healthy():
        return None
    log = open(log_path or os.path.join(ROOT, "results", "logs", "llama_server.log"), "a")
    cmd = [LLAMA_SERVER, "-m", model or MODEL, "-c", str(ctx), "-t", str(threads), "-tb", str(threads), "--port", str(PORT),
           "--host", "127.0.0.1", "-np", "1", "--seed", "2026", "--no-webui",
           "--cache-ram", "512"]                      # bound the host-memory prompt cache (default 8 GB)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if healthy():
            return proc
        time.sleep(1)
    raise RuntimeError("llama-server did not start")


def model_info():
    try:
        with urllib.request.urlopen(URL + "/props", timeout=10) as r:
            p = json.loads(r.read().decode())
        return {"model_path": os.path.basename(p.get("model_path", MODEL)), "n_ctx": p.get("default_generation_settings", {}).get("n_ctx")}
    except Exception:
        return {"model_path": os.path.basename(MODEL)}


_REPLAY = {}


def load_replay(log_file):
    """Load earlier responses from a JSONL log so that an interrupted run resumes without re-generating.
    A stored response is reused only if the tag AND the full prompt are identical."""
    _REPLAY.clear()
    if log_file and os.path.exists(log_file):
        with open(log_file, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _REPLAY[(d.get("tag"), json.dumps(d.get("messages"), sort_keys=True))] = d
    return len(_REPLAY)


def stop_server(proc):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    for _ in range(30):
        if not healthy():
            return
        time.sleep(1)


def chat(messages, max_tokens=256, json_schema=None, top_logprobs=None, log_file=None, tag=None, seed=2026, grammar=None):
    key = (tag, json.dumps(messages, sort_keys=True))
    if key in _REPLAY:
        d = _REPLAY[key]
        out = {"text": d["response"], "seconds": d["seconds"], "usage": d.get("usage", {}), "finish_reason": "replay"}
        if top_logprobs:
            out["top_logprobs"] = [[tuple(x) for x in c] for c in (d.get("top_logprobs") or [])]
        return out
    payload = {"messages": messages, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": seed,
               "max_tokens": max_tokens, "cache_prompt": True}
    if json_schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": json_schema}}
    if grammar is not None:
        payload["grammar"] = grammar
    if top_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(top_logprobs)
    t0 = time.time()
    r = _post("/v1/chat/completions", payload)
    dt = time.time() - t0
    ch = r["choices"][0]
    text = ch["message"].get("content") or ""
    out = {"text": text, "seconds": round(dt, 3), "usage": r.get("usage", {}), "finish_reason": ch.get("finish_reason")}
    if top_logprobs:
        lp = ch.get("logprobs", {}) or {}
        content = lp.get("content") or []
        out["top_logprobs"] = [[(t.get("token"), t.get("logprob")) for t in c.get("top_logprobs", [])] for c in content]
    if log_file:
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": tag, "messages": messages, "max_tokens": max_tokens, "json_schema": json_schema, "grammar": grammar,
                                 "response": text, "seconds": out["seconds"], "usage": out["usage"],
                                 "top_logprobs": out.get("top_logprobs")}, ensure_ascii=False) + "\n")
    return out
