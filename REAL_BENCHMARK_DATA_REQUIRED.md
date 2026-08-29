# Real benchmark data required before manuscript results can be generated

The current execution environment does not contain the authentic benchmark bytes. Do **not** use the smoke-test files or existing smoke result CSVs as manuscript evidence.

## CICIoT2023
Preferred input (official release):

- Place the 169 official CSV files under:
  `data/raw/ciciot2023/csv/`
- Each CSV should contain the official CICIoT2023 flow features and `label` column.

Accepted reproducible alternative (public CC BY 4.0 stratified mirror):

- `train-00000-of-00001.parquet`
- `test-00000-of-00001.parquet`
- `validation-00000-of-00001.parquet`
- Place under `data/raw/ciciot2023/random_3way/`

The alternative contains all 33 attack types grouped into 7 attack families plus Benign and is suitable for a clearly labelled benchmark-on-subsample study.

## Edge-IIoTset
Place the original selected DNN file under:

`data/raw/edgeiiotset/DNN-EdgeIIoT-dataset.csv`

Expected schema: 63 columns including `Attack_label` and `Attack_type`, approximately 2,219,201 rows.

## Required run protocol
Five fixed seeds: 7, 19, 31, 43, 59.
Dirichlet alpha: 1.0, 0.5, 0.3, 0.1 (primary alpha=0.3).
Methods: FedAvg, FedProx, SCAFFOLD, FedAvg+EWC, DEx-FCL.
Zero-day: strict leave-one-family-out; held-out family excluded from train and threshold calibration.

After real files are mounted, run:

```bash
bash scripts/prepare_datasets.sh
bash scripts/run_submission_campaign.sh
```

Then generate tables/figures:

```bash
python -m dexfcl reporting --config configs/ciciot2023.yaml
python -m dexfcl reporting --config configs/edgeiiotset.yaml
```
