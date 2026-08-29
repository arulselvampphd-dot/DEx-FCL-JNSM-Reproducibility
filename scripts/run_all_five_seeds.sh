#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/environment_report.py
python scripts/run_five_seeds.py --config configs/ciciot2023.yaml --aggregate
python scripts/run_five_seeds.py --config configs/edgeiiotset.yaml --aggregate
python -m dexfcl aggregate --results results --tables tables --figures figures
