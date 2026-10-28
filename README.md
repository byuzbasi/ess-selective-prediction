# Social inequalities in prediction coverage across ESS countries and waves

Runnable source code and the two original ESS data releases for a descriptive
study of prediction coverage for observed CES-D8 scores. The analysis compares
CatBoost and TabM in 19 country holdouts, using Round 7 source data and separate
Round 7 and Round 11 target tests. No study results, fitted models, predictions,
logs or manuscript files are stored in this repository.

## Scientific scope

The study examines whether an 80 percent prediction-capacity rule serves social
groups unevenly and how selection changes after calibration on a fixed reserve
of 100 target respondents. The primary target is the equal mean of country-specific
survey-weighted metrics. Results are descriptive. The procedure does not establish
population significance, causal discrimination, clinical utility or superiority
over all simpler supervised models. No bootstrap, confidence intervals or p-values
are produced. The saved reference is a source-fit weighted constant median.

The countries are AT, BE, CH, DE, EE, ES, FI, FR, GB, HU, IE, IL, LT, NL, NO,
PL, PT, SE and SI. Recorded ages 25–89 and eight valid CES-D responses define
the complete-case analysis domain. No outcome imputation, clinical cutoff or
standardization is applied. The 13 predictors, social groups, seeds, source-only
age scaling, missing-input tokens and model hyperparameters are in `config.json`.
There are 31,319 Round 7 and 26,677 Round 11 eligible respondents; 1,900 per wave
are reserved, leaving 29,419 and 24,777 test respondents. Test outcomes are kept
in separate evaluation payloads and are not read by training or selection workers.

## Environment

Use Python **3.13.5**, CPU execution and the exact versions in `requirements.txt`.
Create your own environment using your preferred environment manager, then install
the declared dependencies. `run.py` sets and requires two numerical threads and
uses the currently selected Python interpreter. It has no machine-specific
virtual-environment or architecture launcher. Python source preparation, not R,
reads the raw Stata files. Optional scientific figures require base R.

## Fit-free checks and preparation

Commands below assume this repository was cloned into `ess-selective-prediction`
under the current directory and the recorded Python environment is active.

```bash
cd ess-selective-prediction
python -B run.py check
python -B run.py prepare --run results/run01
```

Preparation creates deterministic cohorts, roles, source-only preprocessing and
133 worker payloads plus 38 separate test-outcome payloads. It starts no learner
and solves no calibrator. Existing run directories are never overwritten.

## User-terminal execution

Run this command yourself only when ready for the numerical workload:

```bash
cd ess-selective-prediction
python -B run.py run --user-terminal --run results/run01
```

A fresh run includes all 19 countries: 38 main model trainings, 114 small
calibrators and 19 constant references (171 estimation units). It does not depend
on previously fitted SI or AT results. If `prepare` was omitted, it is performed
first. The numerical controller uses one worker, two threads, a 60 minute total
allowance and a **sampled** 4 GiB controller-plus-worker RSS threshold. The memory
threshold is not a hard operating-system cap and can be overshot between samples.
Storage and log ceilings are 512 MiB and 16 MiB. Per-country phase allowances are
90 seconds for the reference, 300 for CatBoost, 600 for TabM and 150 for diagnostics.

For context, the preceding measured local analysis completed the 19-country
registry in 448.3 seconds, with 36 newly trained main models and 10 inherited
estimation units. Sampled peak RSS was 0.664 GiB and output storage 152.7 MiB.
These are compatible historical measurements, **not** a timing or numerical
reproduction validation of this fresh-start adapter on another machine. A fresh
38-model run may differ in runtime and floating-point results across platforms.
No new production training was used to validate this public packaging.

## Progress and safe continuation

Progress is flushed at phase boundaries and at least every 30 seconds into stdout,
`progress.json` and append-only `progress.tsv`. Completion is based on validated
countries, not elapsed time divided by the allowance. ETA is stage-specific and
unknown initially. To inspect a running job in another terminal:

```bash
cd ess-selective-prediction
cat results/run01/progress.json
tail -n 10 results/run01/progress.tsv
```

Completed country checkpoints have relative-path SHA-256 manifests. Resume verifies
the source, raw inputs and prepared payloads and preserves the original absolute
deadline. It skips verified complete countries. It **refuses** an incomplete
country instead of silently repeating training; partial files remain available
for inspection. No in-flight training resume or automatic retry is claimed.

```bash
cd ess-selective-prediction
python -B run.py resume --user-terminal --run results/run01
python -B run.py verify --run results/run01
```

## Descriptive evaluation

Only after all 19 countries' predictions and selection decisions are verified:

```bash
cd ess-selective-prediction
python -B run.py evaluate --run results/run01
Rscript src/figures.R results/run01/evaluation
```

Evaluation reads the separate test outcomes, produces country/group metrics,
paired country contrasts, constant-reference comparisons and B0/B100 selection
turnover. CSV tables expose unrounded values; figure scripts use those generated
values. Primary whole-person capacity and source-threshold sensitivity are
different policies and are reported separately. Evaluation does not retrain,
recalibrate or change selection. An existing evaluation is never overwritten.
All generated files remain under ignored `results/`.

## Data and author

The original releases and their separate **CC BY-NC-SA 4.0** terms are documented
in [data/README.md](data/README.md). Source checksums and DOIs are in
`data/sources.json`. Cite both ESS editions when using the data; changes to
derived data must be identified. No ESS access identifier or API credential is needed.

Bahadır Yüzbaşı, Department of Econometrics, İnönü University, Malatya, Türkiye.
ORCID: https://orcid.org/0000-0002-6196-3201
Contact: bahadir.yuzbasi@inonu.edu.tr

Copyright 2026 Bahadır Yüzbaşı. This public source release supplies the analysis
for inspection and scholarly reproduction; it does not alter the ESS data license.
