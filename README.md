# Prediction coverage across European Social Survey countries

Analysis code for a study of educational and migration differences in prediction coverage for CES-D-8 scores across 19 countries. The workflow compares CatBoost and TabM under a common prediction capacity and a fixed local calibration budget.

## Data

| ESS integrated release | Stata file | DOI |
| --- | --- | --- |
| Round 7, edition 2.3 | `data/ESS7e02_3.dta` | [10.21338/ess7e02_3](https://doi.org/10.21338/ess7e02_3) |
| Round 11, edition 4.2 | `data/ESS11e04_2.dta` | [10.21338/ess11e04_2](https://doi.org/10.21338/ess11e04_2) |

Both files are unchanged ESS ERIC releases. See [data attribution and terms](data/README.md) for the CC BY-NC-SA 4.0 license and [data checksums](data/sources.json) for the exact inputs.

## Reproduce the analysis

Use Python **3.13.5**, the pinned [Python dependencies](requirements.txt), and base R for figures. Use a fresh run directory. Clone with `git clone https://github.com/byuzbasi/ess-selective-prediction.git` and run the following in a Bash-compatible shell.

```bash
cd ess-selective-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
python -B run.py check
python -B run.py prepare --run results/run01

# Train the 19 country holdouts when ready.
python -B run.py run --user-terminal --run results/run01

# Verify the completed run and generate tables and figures.
python -B run.py verify --run results/run01
python -B run.py evaluate --run results/run01
Rscript src/figures.R results/run01/evaluation
```

The controller allows 60 minutes and uses a sampled 4 GiB RSS threshold. See `results/run01/progress.json` for progress. `resume` with the same run arguments reuses verified country checkpoints within the original deadline and refuses an incomplete country.

The evaluation directory contains CSV tables, detailed group metrics, and two PDF plots. The country-gap PDF covers education, and the second PDF shows selection turnover. The manuscript uses separately formatted displays. Generated outputs stay in the ignored `results/` directory.

Copyright 2026 Bahadır Yüzbaşı. The analysis code is provided for scholarly reproduction. ESS data retain their separate license.
