$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
python scripts/environment_report.py
python scripts/run_five_seeds.py --config configs/ciciot2023.yaml --aggregate
python scripts/run_five_seeds.py --config configs/edgeiiotset.yaml --aggregate
python -m dexfcl aggregate --results results --tables tables --figures figures
