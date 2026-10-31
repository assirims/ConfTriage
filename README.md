# ConfTriage: risk-controlled large language model assistance for requirements classification and traceability

Replication package for the manuscript submitted to *Expert Systems with Applications*. It contains the raw data, all
code, every prompt and model response, the processed data, all result tables and figures, and a script that checks
every table cell and the key numbers of the manuscript against the result files. Everything runs on an ordinary
computer without a GPU: the language models are 4-bit GGUF files served by llama.cpp on two CPU cores.

## Framework in one paragraph

**ConfTriage** turns the scores of any assistant (open-weight LLMs, lightweight supervised or retrieval models, or a
fusion of both) into decisions with distribution-free guarantees, calibrated on a small labelled sample of requirements:
conformal prediction sets with guaranteed coverage (marginal or class-conditional), automatic labelling with a
guaranteed error budget that holds overall or within every predicted class (conformal risk control; Learn-then-Test as
a stricter variant) and trace link shortlists with a guaranteed expected recall. The evaluation measure is the analyst
effort that remains at a guaranteed risk level.

## Folder layout

| Path | Content |
|---|---|
| `01_verify_raw.py` … `10_verify_manuscript.py` | Pipeline (see `run_all.sh` / `run_all.bat` for the order) |
| `lib/` | Conformal procedures (`conformal.py`), llama.cpp client, ONNX encoders, WordPiece tokenizer, lexical retrieval, metrics |
| `tests/` | Simulation tests of the conformal procedures; tokenizer and metric tests |
| `data/raw/` | Raw data exactly as obtained (`promise_exp/`, `traceability/`) and `MANIFEST.json` with sources and SHA-256 |
| `data/processed/` | Parsed data, embeddings, retrieval scores, LLM probabilities, per-split results |
| `results/llm/` | Every prompt sent to every language model and its response with the top-20 log-probabilities (JSONL). **These stored outputs are the reference from which every table and figure is computed** (see "Determinism" below) |
| `results/tables/` | All result tables (CSV); the mapping to the manuscript tables is given below |
| `results/figures/` | Figures 1–8 (300 dpi PNG), numbered as in the manuscript |
| `results/logs/` | Console logs of the pipeline steps |
| `results/paper_numbers.json` | All result tables and data counts in one JSON file (input of `10_verify_manuscript.py`) |
| `CHANGELOG.md` | Changes since the first version of the package |

## Data

* **PROMISE_exp** (Lima et al., 2019): 969 requirements from 47 projects, 12 classes (CC BY-SA 3.0; derived from the
  PROMISE NFR data set of Cleland-Huang et al., 2007). Source: github.com/AleksandarMitrevski/se-requirements-classification
  (commit 1f5dc45), file `0-datasets/PROMISE_exp/PROMISE_exp.arff`.
* **Traceability**: eTour, iTrust, SMOS (English translation) and LibEST requirements-to-code data sets as distributed with
  the fine-grained traceability benchmark of Hey et al. (2021): github.com/tobhey/finegrained-traceability (commit 197cf2f).
  In SMOS, 17 links of the benchmark name `managerUser.java` while the code folder holds `managerUser.html`;
  `02_build_data.py` matches such links by file stem, so that all 1,044 SMOS links (1,842 links in total) are used.

## Models (not included because of their size, about 5.6 GB)

Qwen2.5-0.5B/1.5B/3B-Instruct-GGUF (Q4_K_M), Llama-3.2-3B-Instruct-GGUF (Q4_K_M, bartowski), all-MiniLM-L6-v2,
bge-small-en-v1.5 and ms-marco-MiniLM-L-6-v2 (ONNX, Xenova). `data/raw/MANIFEST.json` lists for every file the
Hugging Face repository, the pinned revision, the path and the SHA-256 checksum. Download the files at the pinned
revisions, place them under `models/` at the paths listed in the manifest and run `python 01_verify_raw.py --verify`,
which checks the checksums. The models are only needed for steps 03 and 04; all analyses run from the stored outputs.

## Requirements

* Python 3.11 or later with the libraries pinned in `requirements.txt` (`pip install -r requirements.txt`). Other
  versions, in particular of scikit-learn, change some results in the third decimal.
* llama.cpp **release b11398** (`llama-server`), built for the CPU; set the environment variable `LLAMA_SERVER` to the
  binary (default: `llama-server` on the PATH). Only needed for step 04.
* Graphviz (`dot`) for Fig. 1 (step 09).

## Reproduce

```
python 01_verify_raw.py --verify     # reassembles split files, checks SHA-256 of data and models
./run_all.sh                         # Linux/macOS (run_all.bat on Windows)
```

The generative-model steps (04) take about 10 hours on two CPU cores and resume from `results/llm/*.jsonl`. All
analyses can be re-run from the stored model outputs without the models:

```
python 05_classification.py && python 05b_per_class.py && python 06_trace.py && python 07_figures.py && python 08_collect_results.py
```

To check a Word manuscript against the result files (every cell of Tables 2–17, the key numbers in the text and the
citations): `python 10_verify_manuscript.py path/to/manuscript.docx`.

## Determinism and reproducibility

* Steps 01, 02, 05, 05b, 06, 07 and 08 are deterministic (fixed seeds); re-running them on the stored inputs gives
  byte-identical tables.
* The LLM outputs (step 04: greedy decoding, fixed seed, release b11398) are deterministic for a given build and CPU
  but **not across hardware**: the log-probabilities depend on the CPU instruction set, and an independent re-run on
  another CPU changed 2–7% of the classification answers and 0.1–0.3% of the trace answers. The stored responses in
  `results/llm/` are therefore the reference outputs; every table and figure is computed from them.
* The embeddings and retrieval scores (step 03) differ across hardware by at most about 1e-5, which does not change
  the candidate shortlists.
* The logistic regressions on embedding features are sensitive to the library versions (third decimal), hence the pins
  in `requirements.txt`.

## Mapping to the manuscript

| Manuscript | File(s) |
|---|---|
| Table 2 | `data/processed/trace_pairs.csv` (counts) |
| Table 3 | `results/tables/cls_llm_full_sample.csv`; McNemar tests in `cls_llm_mcnemar.csv` (`p`, `p_holm`) |
| Tables 4, 9, 11 | `cls_summary.csv` (mean and SD over 100 splits, project and random design) |
| Table 5 | `cls_per_class.csv` |
| Table 6 | `cls_paired_comparisons.csv` (sign tests over splits; `p_sign`, `p_holm`) |
| Tables 7, 8 | `trace_ranking.csv`, `trace_ranking_shortlist.csv`; Wilcoxon tests in `trace_ranking_tests.csv` (`p_wilcoxon`, `p_holm`) |
| Table 10 | `trace_crc.csv` |
| Table 12 | `cls_calibration_size.csv` |
| Table 13 | `cls_class_coverage.csv` |
| Table 14 | `cls_class_conditional.csv` (per predicted class: `cls_class_conditional_per_class.csv`) |
| Table 15 | `trace_crc_shift.csv` |
| Table 16 | `trace_crc_calsize.csv` (`n_cal` = 10, 20, 30; −1 = the whole calibration half) |
| Table 17 | `data/processed/llm_cls_*.csv`, `llm_trace_*.csv` (column `seconds`) |
| Fig. 1 | `09_make_diagram.py` → `results/figures/Fig1_framework.png` |
| Figs. 2–8 | `07_figures.py` → `results/figures/Fig2_…` to `Fig8_…` |

All P values are reported raw and Holm-adjusted within their family (120 McNemar tests, 56 sign tests, 72 Wilcoxon
tests); the manuscript quotes the Holm-adjusted values.
