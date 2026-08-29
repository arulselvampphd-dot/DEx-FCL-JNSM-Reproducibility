$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
python scripts/environment_report.py
$configs = @("configs/ciciot2023.yaml", "configs/edgeiiotset.yaml")
foreach ($cfg in $configs) {
  python scripts/run_five_seeds.py --config $cfg --protocols closed noniid continual zeroday drift explanation fewshot
  python -m dexfcl run --config $cfg --protocol communication --seed 7
  python scripts/run_ablation.py --config $cfg
  python scripts/run_sensitivity.py --config $cfg
}
python -m dexfcl aggregate --results results --tables tables --figures figures
