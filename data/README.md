# Dataset

This project uses **CICIoT2023**, published by the Canadian Institute for
Cybersecurity (CIC):

https://www.unb.ca/cic/datasets/iotdataset-2023.html

The raw dataset is **not included** in this repository (too large for
GitHub). After downloading, place the CSV splits at the repository root
so the scripts can find them:

```
CICIOT2023/
├── train/
│   └── train.csv
├── test/
│   └── test.csv
└── validation/
    └── validation.csv
```

All scripts under `scripts/` and `figures/` expect to be run from the
repository root (e.g. `python scripts/02_eda.py`), so that the relative
paths `CICIOT2023/...`, `reports/...`, and `models/...` resolve correctly.
