"""Step 01 - raw data and model files: provenance and SHA-256 verification.

Raw data (all public):
  * PROMISE_exp (Lima et al., 2019; CC BY-SA 3.0, derived from the PROMISE NFR data set), file
    0-datasets/PROMISE_exp/PROMISE_exp.arff of https://github.com/AleksandarMitrevski/se-requirements-classification
    (commit 1f5dc4501a1956f21011083594d925bac49f198c)
  * eTour, iTrust, SMOS (English translation) and LibEST requirements-to-code traceability data sets, folders
    datasets/{eTour,iTrust,smos,LibEST} of https://github.com/tobhey/finegrained-traceability
    (commit 197cf2f395e9e90636436f30d6383465cb9e9f63; Hey et al., 2021; GPL-3.0 repository)
Model files (not in the package because of their size; download URLs pinned to repository revisions):
  Qwen2.5-0.5B/1.5B/3B-Instruct-GGUF (Q4_K_M), Llama-3.2-3B-Instruct-GGUF (Q4_K_M, bartowski),
  all-MiniLM-L6-v2, bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2 (ONNX, Xenova)

With the cloned repositories present (data/raw/promise_exp_repo, data/raw/fgt), the script copies the files that are
used into data/raw/promise_exp and data/raw/traceability and writes data/raw/MANIFEST.json. Otherwise it reassembles
split files (*.partNN) and checks every file against MANIFEST.json.
"""
import glob
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
MAN = os.path.join(RAW, "MANIFEST.json")
PROMISE_REPO = ("https://github.com/AleksandarMitrevski/se-requirements-classification", "1f5dc4501a1956f21011083594d925bac49f198c")
FGT_REPO = ("https://github.com/tobhey/finegrained-traceability", "197cf2f395e9e90636436f30d6383465cb9e9f63")
TRACE_DIRS = {"eTour": ["req", "code", "etour_solution_links_english.txt", "README.txt"],
              "iTrust": ["req", "code", "itrust_solution_links.txt", "README.txt"],
              "smos": ["trans_req", "trans_code", "smos_solution_links_italian.txt", "smos_solution_links_english.txt", "README.txt"],
              "LibEST": ["req", "code", "req_to_code_ground.txt"]}
MODELS = {
    "models/qwen2.5-0.5b-instruct-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf": ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf", "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"),
    "models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf": ("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf", None),
    "models/qwen2.5-3b-instruct-gguf/qwen2.5-3b-instruct-q4_k_m.gguf": ("https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/7dabda4d13d513e3e842b20f0d435c732f172cbe/qwen2.5-3b-instruct-q4_k_m.gguf", "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"),
    "models/llama-3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf": ("https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf", "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff"),
    "models/all-MiniLM-L6-v2/onnx/model.onnx": ("https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx", None),
    "models/bge-small-en-v1.5/onnx/model.onnx": ("https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/ea104dacec62c0de699686887e3f920caeb4f3e3/onnx/model.onnx", None),
    "models/ms-marco-MiniLM-L-6-v2/onnx/model.onnx": ("https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/a09144355adeed5f58c8ed011d209bf8ee5a1fec/onnx/model.onnx", None),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    files = {}
    src = os.path.join(RAW, "promise_exp_repo", "0-datasets", "PROMISE_exp", "PROMISE_exp.arff")
    dst = os.path.join(RAW, "promise_exp", "PROMISE_exp.arff")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    files[os.path.relpath(dst, ROOT)] = {"source": f"{PROMISE_REPO[0]} @ {PROMISE_REPO[1]}: 0-datasets/PROMISE_exp/PROMISE_exp.arff"}
    for ds, items in TRACE_DIRS.items():
        for it in items:
            s = os.path.join(RAW, "fgt", "datasets", ds, it)
            d = os.path.join(RAW, "traceability", ds, it)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
                for r, _, fs in os.walk(d):
                    for fn in fs:
                        p = os.path.join(r, fn)
                        files[os.path.relpath(p, ROOT)] = {"source": f"{FGT_REPO[0]} @ {FGT_REPO[1]}: datasets/{ds}/{os.path.relpath(p, os.path.join(RAW, 'traceability', ds))}"}
            elif os.path.exists(s):
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
                files[os.path.relpath(d, ROOT)] = {"source": f"{FGT_REPO[0]} @ {FGT_REPO[1]}: datasets/{ds}/{it}"}
    for rel in files:
        files[rel]["sha256"] = sha256(os.path.join(ROOT, rel))
    models = {}
    for rel, (url, sha) in MODELS.items():
        p = os.path.join(ROOT, rel)
        models[rel] = {"url": url, "sha256": sha256(p) if os.path.exists(p) else sha}
        if sha and os.path.exists(p) and models[rel]["sha256"] != sha:
            raise RuntimeError("model checksum differs from the published one: " + rel)
    json.dump({"data_files": files, "models": models, "note": "data/raw/MANIFEST.json written by 01_verify_raw.py"},
              open(MAN, "w"), indent=1)
    print(f"collected {len(files)} raw data files; manifest written")


def reassemble():
    for first in glob.glob(os.path.join(ROOT, "**", "*.part01"), recursive=True):
        target = first[: -len(".part01")]
        if os.path.exists(target):
            continue
        with open(target, "wb") as out:
            for p in sorted(glob.glob(target + ".part[0-9][0-9]")):
                with open(p, "rb") as fh:
                    out.write(fh.read())
        print("reassembled", os.path.relpath(target, ROOT))


def verify():
    reassemble()
    man = json.load(open(MAN))
    ok = bad = missing = 0
    for group in ("data_files", "models"):
        for rel, info in man[group].items():
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                missing += 1
                print("not present:", rel, ("- download from " + info["url"]) if "url" in info else "")
                continue
            if info.get("sha256") and sha256(p) != info["sha256"]:
                bad += 1
                print("CHECKSUM MISMATCH:", rel)
            else:
                ok += 1
    print(f"verified {ok} files, {bad} mismatches, {missing} not present")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    if os.path.isdir(os.path.join(RAW, "fgt")) and os.path.isdir(os.path.join(RAW, "promise_exp_repo")) and "--verify" not in sys.argv:
        collect()
    verify()
