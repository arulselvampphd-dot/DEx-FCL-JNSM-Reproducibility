# Dataset acquisition and non-redistribution

> These are historical paper acquisition notes. The legacy preparation launcher
> does not implement the supplied Parquet splits or integrate the Edge 10% subset.
> See RELEASE_STATUS.md before attempting a training reproduction. Exact raw
> dataset distribution hashes have not been recovered.

Raw benchmark data are **not redistributed** in this repository.

## CICIoT2023

Authoritative dataset paper/source:
- E. C. P. Neto et al., “CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment,” Sensors 23(13), 5941 (2023), DOI: 10.3390/s23135941.
- Canadian Institute for Cybersecurity, University of New Brunswick dataset portal: https://www.unb.ca/cic/datasets/iotdataset-2023.html

The manuscript experiments used the public `random_3way` Parquet release with the supplied train/validation/test splits. The repository does not redistribute those Parquet files. Place them under `data/raw/ciciot2023/` using the names documented in `data/README.md`.

## Edge-IIoTset

Authoritative dataset paper/source:
- M. A. Ferrag et al., “Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning,” IEEE Access 10, 40281–40306 (2022), DOI: 10.1109/ACCESS.2022.3165809.
- Dataset author's Kaggle distribution: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot

Download `DNN-EdgeIIoT-dataset.csv`, then create the exact 10% subset used in the manuscript:

```bash
python scripts/make_edgeiiot_10pct.py DNN-EdgeIIoT-dataset.csv
```

The resulting file is `DNN-EdgeIIoT-10pct-seed42.csv` and uses exact class-stratified sampling by `Attack_type` with seed 42.
