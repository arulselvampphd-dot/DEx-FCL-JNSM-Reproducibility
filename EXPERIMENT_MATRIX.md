# Submission experiment matrix

All primary experiments use seeds **7, 19, 31, 43, 59**.

| Experiment | CICIoT2023 | Edge-IIoTset | Output |
|---|---|---|---|
| Closed-set FL | FedAvg, FedProx, SCAFFOLD, FedAvg+EWC, DEx-FCL | same | `closed_*.csv` |
| Non-IID sensitivity | α=1.0,0.5,0.3,0.1 | same | `noniid_*.csv` + partition manifests |
| Continual learning | 6 sequential grouped tasks | 5 sequential grouped tasks | `continual_*.csv`, `continual_stages_*.csv` |
| Zero-day | leave one attack family out; threshold calibrated on known validation only | same | `zeroday_*.csv`, raw score NPZ files |
| Statistical analysis | Friedman + paired Wilcoxon | same | `table_statistics.csv` |
| Figures | non-IID, forgetting, zero-day | same | PNG + PDF |

## Continual sequences

CICIoT2023:
1. Benign + DDoS + DoS
2. Recon
3. Spoofing
4. BruteForce
5. WebBased
6. Mirai

Edge-IIoTset:
1. Normal + DDoS
2. Scanning
3. Injection
4. Malware
5. MITM

The order should also be randomized in an additional robustness analysis if computational budget permits; the fixed sequence above is the preregistered primary sequence.
