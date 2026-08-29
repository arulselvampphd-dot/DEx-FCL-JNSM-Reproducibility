#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/environment_report.py
for CFG in configs/ciciot2023.yaml configs/edgeiiotset.yaml; do
  python scripts/run_five_seeds.py --config "$CFG" --protocols closed noniid continual zeroday drift explanation fewshot
  python -m dexfcl run --config "$CFG" --protocol communication --seed 7
  python scripts/run_ablation.py --config "$CFG"
  python scripts/run_sensitivity.py --config "$CFG"
done
python -m dexfcl aggregate --results results --tables tables --figures figures
