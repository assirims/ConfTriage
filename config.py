"""Shared settings for the study 'Enhancing software requirements engineering through large language model assistance'."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")
LOGS = os.path.join(RESULTS, "logs")
LLM_LOGS = os.path.join(RESULTS, "llm")
MODELS = os.path.join(ROOT, "models")
for _d in (PROC, TABLES, FIGS, LOGS, LLM_LOGS):
    os.makedirs(_d, exist_ok=True)

SEED = 2026
N_BOOT = 2000          # bootstrap replicates
N_SPLITS = 500         # random calibration/test splits for conformal evaluation
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10           # Learn-then-Test: guarantee holds with probability >= 1 - DELTA

# generative models (GGUF, 4-bit Q4_K_M), all run locally with llama.cpp on CPU
LLMS = {
    "qwen0.5b": os.path.join(MODELS, "qwen2.5-0.5b-instruct-gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "qwen1.5b": os.path.join(MODELS, "qwen2.5-1.5b-instruct-gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "qwen3b": os.path.join(MODELS, "qwen2.5-3b-instruct-gguf", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "llama3b": os.path.join(MODELS, "llama-3.2-3b-instruct-gguf", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
}
LLM_LABEL = {"qwen0.5b": "Qwen2.5-0.5B", "qwen1.5b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B", "llama3b": "Llama-3.2-3B"}

# sentence encoders (ONNX)
MINILM = os.path.join(MODELS, "all-MiniLM-L6-v2")
BGE = os.path.join(MODELS, "bge-small-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CROSS = os.path.join(MODELS, "ms-marco-MiniLM-L-6-v2")

# requirements classification (PROMISE_exp)
CLASSES = ["F", "A", "FT", "L", "LF", "MN", "O", "PE", "PO", "SC", "SE", "US"]
CLASS_NAMES = {"F": "Functional", "A": "Availability", "FT": "Fault tolerance", "L": "Legal", "LF": "Look and feel",
               "MN": "Maintainability", "O": "Operational", "PE": "Performance", "PO": "Portability", "SC": "Scalability",
               "SE": "Security", "US": "Usability"}

# trace link recovery (use cases/requirements -> code classes)
TRACE_SETS = ["eTour", "iTrust", "SMOS", "LibEST"]
TRACE_K = 20           # candidates per requirement verified by the generative model
