# DEx-FCL — archived JNSM numerical outputs

This repository accompanies “DEx-FCL: Drift- and Explanation-Consistent Federated Continual Learning for Adaptive Edge-IoT Security Management.”

**Status: archived numerical results can be regenerated; end-to-end model-training provenance remains unresolved.** See [RELEASE_STATUS.md](RELEASE_STATUS.md) and [SOURCE_RECOVERY.md](SOURCE_RECOVERY.md).

## Reproduce the archived tables and figures

The canonical CSVs under `results/` contain five model seeds: **7, 19, 31, 43, 59**. This workflow validates the experiment matrix and regenerates numerical summaries and all 11 figure names without downloading datasets or training models.

Python 3.12 was used for the numerical-export environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-paper.txt
# Linux/macOS
PYTHONPATH=src python -m dexfcl reproduce-paper
# Windows PowerShell
# $env:PYTHONPATH = "src"
# python -m dexfcl reproduce-paper
```

Outputs go to `artifacts/paper/tables/` and `artifacts/paper/figures/`. Figures use the historical names with 600-dpi PNG, PDF and EPS exports. The layout is newly regenerated; historical figure pixels are not expected to match. A JSON manifest records input/output SHA-256 hashes, row counts, software versions and missing timing/delay observations. Archived files remain intact.

The original aggregation command now accepts the canonical archived schema:

```bash
PYTHONPATH=src python -m dexfcl aggregate --source archived --results results --tables artifacts/paper/tables --figures artifacts/paper/figures
```

`reproduce-paper` additionally writes the provenance manifest. When archived and new campaign results coexist, automatic aggregation stops and requires an explicit `--source`. Campaign mode selects only named per-dataset/per-seed exports rather than pooling overlapping archived files.

## Included numerical evidence

| Canonical file | Rows | Coverage |
| --- | ---: | --- |
| primary_runs.csv | 50 | 2 datasets × 5 methods × 5 seeds |
| noniid_runs.csv | 160 | 2 datasets × 4 methods × 4 alphas × 5 seeds |
| continual_runs.csv | 30 | 2 datasets × 3 methods × 5 seeds |
| zero_day_runs.csv | 60 | 12 held-out families × 5 seeds; four detector columns |
| assimilation_runs.csv | 50 | 2 datasets × 5 support counts × 5 seeds |
| drift_runs.csv | 30 | 2 datasets × 3 signals × 5 seeds |
| explanation_runs.csv | 20 | 2 datasets × 2 methods × 5 seeds |
| ablation_all_runs.csv | 50 | 2 datasets × 5 removal configurations × 5 seeds |

Only `ablation_all_runs.csv` is pooled for the archived ablation analysis; component exports are excluded. Sample standard deviations use ddof=1. Missing delay for undetected drift is retained as NA, not interpreted as zero delay. Zero-day timing is unavailable in the historical export.

## Dataset acquisition and sampling

Raw datasets are not redistributed. See [DATASETS.md](DATASETS.md) for historical sources.

```bash
python scripts/make_edgeiiot_10pct.py DNN-EdgeIIoT-dataset.csv --output data/raw/edgeiiotset/DNN-EdgeIIoT-10pct-seed42.csv
```

The sampler writes a sidecar manifest with source/output hashes and per-class counts. Sampling depends on the source file's row order. Per-class targets use `max(1, round(n * 0.10))`; this is not necessarily exactly 10% globally because of rounding.

## Training implementation and release limits

`src/dexfcl/`, the dataset YAMLs and training launchers are a legacy implementation. They do not match the paper's recorded random_3way Parquet preprocessing, fixed evaluation splits, five-client model configuration, drift protocol and removal ablations. Changing parameters alone would not recover the generating code.

The separately prepared JSC rerun overlay and Colab notebook are a new experimental track, not provenance for the archived JNSM CSVs. They have not been imported here.

**Do not describe this branch as a verified training reproduction or use its legacy launchers to validate the historical results.** Release acceptance requires recovering the exact generating pipeline or running and documenting a new benchmark campaign, then reconciling the paper.

## Checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
sha256sum --check SHA256SUMS.txt
```

CI runs numerical validation/export and checksum verification. The numerical dependency lock is not a claim about the original training environment. Code licensing remains unresolved; no new license is assumed.
