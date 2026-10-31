# Changes to the replication package (2026-10-31)

This version brings the package in line with the revised manuscript. Nothing in the raw data, the prompts or the
stored model outputs (`results/llm/`) changed.

## Data

* `02_build_data.py`: trace links whose target file name is not found are matched by file stem. In SMOS, 17 links of
  the benchmark name `managerUser.java` while the code folder holds `managerUser.html`; they were silently dropped
  before. SMOS now has 1,044 gold links (1,027 before) and the four systems 1,842 (1,825 before).
  `data/processed/trace_links.csv` and the `gold` column of `data/processed/trace_pairs.csv` were updated accordingly
  (the retrieval scores and embeddings in `trace_pairs.csv` are unchanged; they do not depend on the links).

## Analyses

* `05_classification.py`: class-conditional conformal risk control for automatic labelling (one threshold per
  predicted class, so that the error budget holds within every predicted class) is computed for every method, split
  and budget. New outputs: `results/tables/cls_class_conditional.csv` (Table 14) and
  `cls_class_conditional_per_class.csv` (counts per predicted class pooled over the splits; Fig. 7). `cls_summary.csv`
  gained the columns `cc12_*` and `cc2_*`.
* `06_trace.py`: recall control calibrated on only the first 10, 20 or 30 requirements with at least one gold link of
  the calibration half of the same 500 splits (seed 2033 = SEED + 7). New output: `results/tables/trace_crc_calsize.csv`
  (Table 16, Fig. 8; `n_cal = -1` is the whole calibration half).
* Holm adjustment: `cls_llm_mcnemar.csv`, `cls_paired_comparisons.csv` and `trace_ranking_tests.csv` carry a
  `p_holm` column (Holm-adjusted within the family of 120 McNemar, 56 sign and 72 Wilcoxon tests). The manuscript
  quotes the Holm-adjusted values.
* `07_figures.py`: figure numbers follow the manuscript. New `Fig7_class_conditional_risk.png` (marginal versus
  class-conditional risk control per predicted class) and `Fig8_trace_validity.png` (realised recall when calibrated
  on half of the system, on 20 requirements of the system or on the other systems). The old `Fig7_trace_validity.png`
  (without the 20-requirement markers) was removed.
* `09_make_diagram.py` (Fig. 1): "TF-IDF" spelt correctly; box 3 names the marginal or class-conditional prediction
  sets, the error budget that holds overall or within every predicted class, and recall-controlled link vetting.
* `08_collect_results.py`: the new tables are included in `results/paper_numbers.json`.
* `10_verify_manuscript.py`: now recomputes every cell of Tables 2–17 from the result files (with the paper's
  rounding), searches the key numbers of the text and checks the citations. Usage:
  `python 10_verify_manuscript.py path/to/manuscript.docx`.

## Regenerated results

* All tables in `results/tables/`, `results/paper_numbers.json` and Figures 1–8 were regenerated from the stored
  model outputs with the versions pinned in `requirements.txt` (Python 3.15, numpy 2.5.4, pandas 2.2.3, scipy 1.18.1,
  scikit-learn 1.9.1, matplotlib 3.11.2) on an Apple M4.
* The tracing results (Tables 2, 7, 8, 10, 15, 16; Figs. 5 and 8) changed because of the 17 SMOS links; the
  regenerated values are the ones in the revised manuscript.
* The classification results (Tables 3–6, 9, 11–14; Figs. 2–4, 6, 7) do not depend on the link fix, but the
  logistic regressions on embedding features (BGE + LR, MiniLM + LR and the fusions) are sensitive to the CPU/BLAS
  library, so some cells differ in the last printed digit from the tables of the first package version (which were
  produced on the authors' virtual machine). On the Apple M4, scikit-learn 1.5.2, 1.6.1, 1.7.2, 1.8.0 and 1.9.1 gave
  identical results to each other, so the differences come from the hardware, not from the library version. `10_verify_manuscript.py` run on the revised manuscript (`results/logs/10_verify.log`): 1,750 table cells checked,
  185 differ, all of them in the BGE + LR, MiniLM + LR and fusion rows of Tables 4, 6, 9, 11, 12, 13 and 14, by one
  unit of the last printed digit, except the marginal worst-class risk of Llama-3.2-3B + BGE fusion in Table 14
  (14.2% in the manuscript, 12.8% regenerated; portability, about 226 pooled predictions) and one significance mark
  in Table 6 (Llama-3.2-3B + BGE fusion versus the LLM alone, automation at a 5% budget: +3.1 (62/36) in the
  manuscript, +3.4 (65/34)† regenerated). Tables 2, 3, 5, 7, 8, 10, 15, 16 and 17 match completely. To make the
  package and the manuscript agree exactly, either re-run `05_classification.py` (then `07`, `08`) on the original
  machine with the versions used there and pin those versions in `requirements.txt`, or update the listed manuscript
  cells from the regenerated tables.

## Housekeeping

* README rewritten (current title, no references to folders that are not shipped, model download step, llama.cpp
  release, determinism caveat, mapping of every table and figure to its file).
* `run_all.sh` / `run_all.bat`: the two final steps that referred to files outside the package were removed; the
  manuscript check is run only if a manuscript is present under `manuscript/`.
* `requirements.txt`: exact versions.
* `config.py`, `lib/lexical.py`, `lib/metrics.py`: docstrings corrected; `lib/llm.py`: the default llama-server path
  is now `llama-server` (PATH) instead of a developer-specific path.
* `P-1.zip` (a 22 MB copy of the package inside the package) was removed.
* `results/logs/03_features.log` was from an earlier run with K = 10 and was removed; it will be rewritten by step 03
  when the pipeline is run from the raw data with the models present. The logs of steps 05, 06 and 10 were rewritten.
