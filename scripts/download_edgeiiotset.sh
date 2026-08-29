#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/edgeiiotset
cat <<'MSG'
This command uses the Kaggle dataset mirror. It requires Kaggle credentials.
Official paper: Ferrag et al., IEEE Access 2022, DOI 10.1109/ACCESS.2022.3165809.
MSG
kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot \
  -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv" \
  -p data/raw/edgeiiotset --unzip
