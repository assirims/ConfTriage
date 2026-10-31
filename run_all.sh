#!/bin/bash
# Full pipeline. Steps 04 (generative models) take about 10 hours on two CPU cores; they resume from results/llm/*.jsonl.
# The analyses (05-10) can be re-run from the stored model outputs without the models (see README).
set -e
cd "$(dirname "$0")"
python3 01_verify_raw.py --verify
python3 02_build_data.py
python3 03_features.py
for m in qwen0.5b qwen1.5b qwen3b llama3b; do python3 04_llm_runs.py $m cls; done
for m in qwen1.5b qwen3b; do python3 04_llm_runs.py $m trace; done   # Qwen2.5-0.5B and Llama-3.2-3B: classification only
python3 05_classification.py
python3 05b_per_class.py
python3 06_trace.py
python3 07_figures.py
python3 09_make_diagram.py
python3 08_collect_results.py
python3 tests/test_conformal.py
python3 tests/test_wordpiece_metrics.py
# optional: check a Word manuscript against the result files (every table cell, key numbers, citations)
if [ -f manuscript/RELLM_Manuscript.docx ]; then python3 10_verify_manuscript.py manuscript/RELLM_Manuscript.docx; fi
