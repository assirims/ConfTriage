@echo off
rem Full pipeline (Windows). The generative-model steps take about 10-12 hours on two CPU cores and resume from results\llm.
cd /d "%~dp0"
python 01_verify_raw.py --verify || goto :error
for %%s in (02_build_data 03_features) do ( python %%s.py || goto :error )
for %%m in (qwen0.5b qwen1.5b qwen3b llama3b) do ( python 04_llm_runs.py %%m cls || goto :error )
for %%m in (qwen1.5b qwen3b) do ( python 04_llm_runs.py %%m trace || goto :error )
for %%s in (05_classification 05b_per_class 06_trace 07_figures 09_make_diagram 08_collect_results) do ( python %%s.py || goto :error )
python tests\test_conformal.py
python tests\test_wordpiece_metrics.py
node manuscript\build_manuscript.js
python 10_verify_manuscript.py
echo Done.
pause
exit /b 0
:error
echo A step failed.
pause
