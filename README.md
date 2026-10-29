# Enhancing software requirements engineering through large language model assistance

Replication package for the manuscript submitted to *Expert Systems with Applications*. It contains the raw data, all
code, every prompt and model response, the processed data, result tables, figures and the manuscript. Everything runs
on an ordinary computer without a GPU: the language models are 4-bit GGUF files served by llama.cpp on two CPU cores.

## Framework in one paragraph

**ConfTriage** turns the scores of any assistant (open-weight LLMs, lightweight supervised or retrieval models, or a
fusion of both) into decisions with distribution-free guarantees, calibrated on a small labelled sample of requirements:
conformal prediction sets with guaranteed coverage, automatic labelling with a guaranteed error budget (conformal risk
control; Learn-then-Test as a stricter variant) and trace link shortlists with a guaranteed expected recall. The
evaluation measure is the analyst effort that remains at a guaranteed risk level.

## Folder layout

| Path | Content |
|---|---|
| `00_PAPER/` | Manuscript (Word and PDF, with Appendix A prompts and Appendix B supplementary tables), cover letter, submission checklist |
| `01_verify_raw.py` … `10_verify_manuscript.py` | Pipeline (see `run_all.sh` / `run_all.bat` for the order) |
| `lib/` | Conformal procedures (`conformal.py`), llama.cpp client, ONNX encoders, WordPiece tokenizer, lexical retrieval, metrics |
| `tests/` | Simulation tests of the conformal procedures; tokenizer and metric tests |
| `data/raw/` | Raw data exactly as obtained (`promise_exp/`, `traceability/`) and `MANIFEST.json` with sources and SHA-256 |
| `data/processed/` | Parsed data, embeddings, retrieval scores, LLM probabilities, per-split results |
| `results/llm/` | Every prompt sent to every language model and its response with the top-20 log-probabilities (JSONL) |
| `results/tables/` | All result tables (CSV) |
| `results/figures/` | Figures 1–7 (300 dpi PNG) |
| `results/logs/` | Console logs |
| `results/paper_numbers.json` | Every number quoted in the manuscript |
| `manuscript/` | Node.js scripts that build the manuscript from `results/paper_numbers.json` |

## Data

* **PROMISE_exp** (Lima et al., 2019): 969 requirements from 47 projects, 12 classes (CC BY-SA 3.0; derived from the
  PROMISE NFR data set of Cleland-Huang et al., 2007). Source: github.com/AleksandarMitrevski/se-requirements-classification
  (commit 1f5dc45), file `0-datasets/PROMISE_exp/PROMISE_exp.arff`.
* **Traceability**: eTour, iTrust, SMOS (English translation) and LibEST requirements-to-code data sets as distributed with
  the fine-grained traceability benchmark of Hey et al. (2021): github.com/tobhey/finegrained-traceability (commit 197cf2f).

## Models (not included because of their size; URLs pinned to revisions and SHA-256 in `data/raw/MANIFEST.json`)

Qwen2.5-0.5B/1.5B/3B-Instruct-GGUF (Q4_K_M), Llama-3.2-3B-Instruct-GGUF (Q4_K_M, bartowski), all-MiniLM-L6-v2,
bge-small-en-v1.5 and ms-marco-MiniLM-L-6-v2 (ONNX, Xenova). Place them under `models/` with the paths listed in the manifest.

## Requirements

Python 3.11+ with numpy, pandas, scipy, scikit-learn, onnxruntime, matplotlib, pillow, python-docx (`requirements.txt`);
llama.cpp (`llama-server`, release b11398 or later; set `LLAMA_SERVER`); Graphviz (`dot`); Node.js 18+ with `docx`.

## Reproduce

```
python 01_verify_raw.py --verify     # reassembles split files, checks SHA-256 of data and models
./run_all.sh                         # Linux/macOS (run_all.bat on Windows)
```

The generative-model steps take about 10 hours on two CPU cores. They are deterministic (greedy decoding, fixed seed) and
resume from `results/llm/*.jsonl`, so the analyses can be re-run from the stored model outputs without the models:
`python 05_classification.py && python 05b_per_class.py && python 06_trace.py && python 07_figures.py && python 08_collect_results.py`.
