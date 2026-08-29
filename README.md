# DEx-FCL — real-benchmark reproducibility package

This repository accompanies the manuscript **“DEx-FCL: Drift- and Explanation-Consistent Federated Continual Learning for Adaptive Edge-IoT Security Management.”**

It contains the executable DEx-FCL implementation, deterministic federated partitioning utilities, five-seed experiment configuration, real CICIoT2023 and Edge-IIoTset result CSVs, generated manuscript figures, and the exact Edge-IIoTset 10% sampling script. Raw benchmark datasets are intentionally not redistributed.

## Reproduced experiments

- Five-seed closed-set comparison: FedAvg, FedProx, SCAFFOLD, FedAvg+EWC, DEx-FCL
- Dirichlet non-IID analysis: α ∈ {1.0, 0.5, 0.3, 0.1}
- Federated continual learning and catastrophic forgetting
- Genuine leave-one-attack-family-out zero-day evaluation
- Few-shot unknown-to-known assimilation
- Natural family-transition drift detection
- Cross-client explanation consistency
- Communication accounting
- Five-seed ablation and non-parametric statistical testing

Fixed model seeds: **7, 19, 31, 43, 59**.

## Repository structure

```text
src/dexfcl/              DEx-FCL implementation
scripts/                 experiment launchers and dataset utilities
configs/                 dataset and submission configurations
results/                 raw and summarized real benchmark outputs
results/figures/         publication figures
data/README.md            expected local raw-data layout
DATASETS.md               authoritative dataset acquisition instructions
RESULTS_SUMMARY.md        mapping of result files to manuscript analyses
EXPERIMENT_MATRIX.md      experimental matrix
MANUSCRIPT_INTEGRATION.md mapping from outputs to manuscript sections
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

## Dataset preparation

See [`DATASETS.md`](DATASETS.md). Raw CICIoT2023 and Edge-IIoTset files are not included due to size and redistribution constraints.

For Edge-IIoTset, after downloading the official DNN CSV:

```bash
python scripts/make_edgeiiot_10pct.py DNN-EdgeIIoT-dataset.csv
```

## Five-seed execution

The package includes campaign launchers under `scripts/`. The fixed submission protocol is recorded in `configs/submission_protocol.yaml`.

Typical commands:

```bash
python scripts/run_five_seeds.py --config configs/ciciot2023.yaml
python scripts/run_five_seeds.py --config configs/edgeiiotset.yaml
```

Use `scripts/run_submission_campaign.sh` or the PowerShell equivalent for the complete campaign. See `EXPERIMENT_MATRIX.md` for all experiment types.

## Results already included

The result CSVs and manuscript figures from the real benchmark campaign are committed under `results/`; see `RESULTS_SUMMARY.md` for the file map. These outputs were generated from CICIoT2023 and the deterministic Edge-IIoTset 10% subset, not from synthetic data.

## Reproducibility notes

- CICIoT2023 validation and test prevalence were preserved; only the training split was capped by family for computational tractability.
- Edge-IIoTset used exact 10% `Attack_type`-stratified sampling with seed 42, then a fixed 70:10:20 split; training-only family capping was used while validation/test prevalence was preserved.
- Model seeds: 7, 19, 31, 43, 59.
- Primary federated configuration: five clients and Dirichlet α=0.3; heterogeneity analysis uses α=1.0, 0.5, 0.3, 0.1.
- Raw benchmark data are never committed to this repository.

## Data and code availability

All code, deterministic preprocessing/partition scripts, fixed configurations, real result CSVs and generated figures required to audit the reported numerical results are included here. The original benchmark datasets must be obtained from their authoritative distributors; see `DATASETS.md`.

## Responsible-use note

This repository is provided for reproducible cybersecurity research and defensive intrusion-detection evaluation.
