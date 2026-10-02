# Data placement

> This layout belongs to the legacy implementation and differs from the paper's
> random_3way Parquet protocol. It is not a verified historical training recipe.
> See ../RELEASE_STATUS.md and ../DATASETS.md for the outstanding preprocessing gaps.

Raw datasets are intentionally **not redistributed** in this package.

## CICIoT2023
Download the official CSV release from the Canadian Institute for Cybersecurity (UNB) dataset page. Place the extracted 169 `part-*.csv` files under:

`data/raw/ciciot2023/csv/`

The official release contains 33 attacks grouped into DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, and Mirai, plus benign traffic. The package maps these to eight grouped classes.

## Edge-IIoTset
Use the publisher dataset or the Kaggle mirror and place:

`DNN-EdgeIIoT-dataset.csv`

under `data/raw/edgeiiotset/`.

The package maps the 14 attack types to DDoS, Scanning/Information Gathering, MITM, Injection, and Malware, plus Normal.

## Important preprocessing decision
The package uses numeric features only and removes obvious endpoint identifiers, timestamps, payload/content fields, and label columns. This is deliberate to reduce feature leakage, especially for Edge-IIoTset.
