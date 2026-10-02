# Real benchmark results

> Archived numerical outputs are validated and regenerated with
> `python -m dexfcl reproduce-paper` (with `PYTHONPATH=src`). Their generating
> training source/configuration has not been recovered; see RELEASE_STATUS.md.

This directory contains the machine-readable outputs used to generate the manuscript tables and figures. No synthetic result is used in the final benchmark manuscript.

Primary files:
- `results/primary_runs.csv` — five-seed closed-set comparisons.
- `results/noniid_runs.csv` — Dirichlet heterogeneity analysis.
- `results/continual_runs.csv` — continual-learning and forgetting measurements.
- `results/zero_day_runs.csv` — genuine leave-one-family-out open-set evaluation.
- `results/assimilation_runs.csv` — few-shot unknown-to-known assimilation.
- `results/drift_runs.csv` — natural attack-family transition drift tests.
- `results/explanation_runs.csv` — cross-client explanation consistency.
- `results/ablation_all_runs.csv` — five-seed ablation campaign.
- `results/table_*.csv` — summary tables used in the manuscript.
- `results/figures/*.png` — publication figures regenerated from result files.

The five model seeds are 7, 19, 31, 43 and 59. Dataset-construction seed is 42 where applicable.
