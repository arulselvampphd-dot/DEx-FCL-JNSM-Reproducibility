#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
# CICIoT2023: put the official CSV/ directory (169 part-*.csv files) at data/raw/ciciot2023/csv
python -m dexfcl prepare --dataset ciciot2023 --input data/raw/ciciot2023/csv --output data/processed/ciciot2023_grouped.npz --cap 120000 --seed 2026
# Edge-IIoTset: put DNN-EdgeIIoT-dataset.csv anywhere under data/raw/edgeiiotset
python -m dexfcl prepare --dataset edgeiiotset --input data/raw/edgeiiotset --output data/processed/edgeiiotset_grouped.npz --cap 120000 --seed 2026
