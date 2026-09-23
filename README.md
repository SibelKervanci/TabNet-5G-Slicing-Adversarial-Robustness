# Slice-Aware Adversarial Vulnerability Analysis of TabNet-Based Intrusion Detection in 5G Network Slicing

> **Paper:** Gözde Özsert Yiğit & Ilkay Sibel Kervanci — *Symmetry* (MDPI), manuscript ID: symmetry-4551760  
> **Dataset:** [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) — 34 attack classes, 46 features, 3 5G network slices (eMBB / URLLC / mMTC)

---

## Repository Structure

```
ciciot2023_analysis/
│
├── 01_eda.py                        # Exploratory data analysis, class distribution
├── 02_preprocessing.py              # Feature engineering, label encoding, train/test split
├── 03_tabnet_baseline.py            # TabNet training, confusion matrix, per-class F1
│                                    #   → outputs: models/tabnet_model.*
│                                    #   → outputs: reports/top_attention_features.csv  ← required by 04c
├── 04a_fgsm_attack.py               # FGSM adversarial evaluation
├── 04b_pgd_attack.py                # PGD adversarial evaluation
├── 04c_age_transfer.py              # AGE-Attention attack (surrogate MLP + TabNet attention)
│                                    #   ← requires: reports/top_attention_features.csv
├── 05_psri.py                       # Per-Slice Robustness Index (PSRI) computation
├── 10_multiseed_bootstrap.py        # Multi-seed bootstrap confidence intervals
├── 11_deepshap_ablation.py          # DeepSHAP ablation (AGE-SHAP variant)
├── 12_stronger_baselines.py         # C&W L2 + AutoAttack (APGD-CE) baselines
├── 13_topk_sensitivity.py           # Top-K feature sensitivity analysis (K=1,3,5,7,10,15,20)
│
├── fig1_perslice.py                 # Figure 1 — Per-slice F1 vs Global F1 bar chart
├── fig2_slice_dagilim.py            # Figure 2 — Slice distribution pie + attack type bar
├── fig3_attack_flow.py              # Figure 3 — AGE attack flow diagram
├── fig_confusion_matrix.py          # Confusion matrix heatmap
│
├── models/                          # Saved TabNet model weights
└── reports/                         # All output figures, tables, CSV files
    ├── top_attention_features.csv   # Top-5 attention-ranked features (required by 04c)
    ├── topk_sensitivity.csv         # Top-K sensitivity results
    └── *.png                        # All figures (300 DPI)
```

---

## Requirements

```bash
pip install pytorch-tabnet torch scikit-learn pandas numpy matplotlib seaborn shap
pip install git+https://github.com/fra31/auto-attack   # AutoAttack (not on PyPI)
```

Python ≥ 3.9, PyTorch ≥ 1.12 recommended.

---

## Script Execution Order

Run scripts **in order**. Each script depends on outputs from the previous ones.

### Step 1 — Preprocessing & Baseline

```bash
python 01_eda.py
python 02_preprocessing.py
python 03_tabnet_baseline.py
```

> `03_tabnet_baseline.py` saves the trained TabNet model to `models/` and writes `reports/top_attention_features.csv` (top-5 attention-ranked feature names).  
> **This file must exist before running `04c_age_transfer.py`.**

### Step 2 — Adversarial Attacks

```bash
python 04a_fgsm_attack.py
python 04b_pgd_attack.py
python 04c_age_transfer.py      # requires reports/top_attention_features.csv
```

### Step 3 — PSRI & Slice Analysis

```bash
python 05_psri.py
```

### Step 4 — Additional Experiments (paper revision)

```bash
python 10_multiseed_bootstrap.py    # multi-seed confidence intervals
python 11_deepshap_ablation.py      # AGE-SHAP ablation variant
python 12_stronger_baselines.py     # C&W L2 + AutoAttack baselines
python 13_topk_sensitivity.py       # Top-K sensitivity (K=1,3,5,7,10,15,20)
```

### Step 5 — Figures

```bash
python fig1_perslice.py
python fig2_slice_dagilim.py
python fig3_attack_flow.py
python fig_confusion_matrix.py
```

All figures are saved to `reports/` at 300 DPI.

---

## Key Results

| Attack | Acc Drop | URLLC PSRI | L2 Norm | Modified Features |
|---|---|---|---|---|
| AGE-Attention (proposed) | 50.2% | 0.513 | 0.216 | 4.7 |
| FGSM | 39.0% | 0.412 | 0.664 | 44.1 |
| PGD | 43.3% | 0.580 | 0.473 | 41.5 |
| C&W L2 | 54.8% | 0.023 | 0.235 | 35.5 |
| AutoAttack (APGD-CE) | 57.4% | 0.006 | 2.152 | 30.2 |

**Top-5 attention-ranked features** (from `reports/top_attention_features.csv`):  
`syn_count`, `IAT`, `SMTP`, `TCP`, `Protocol Type`

**Key finding:** AGE-Attention achieves the best effectiveness–stealthiness trade-off. High-power attacks (C&W, AutoAttack) collapse URLLC PSRI to near zero, making them detectable by service-level monitoring in operational 5G networks.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| TabNet n_steps | 5 |
| TabNet n_d / n_a | 64 / 64 |
| Surrogate MLP | 256-128-64, ReLU, Adam, 30 epochs |
| Attack ε | 0.10 |
| AGE top-K | 5 |
| PGD steps | 10 |
| C&W steps | 200 |
| Eval batch size | 500 |
| Random seed | 42 |

---

## Citation

If you use this code or dataset mapping, please cite:

```
Özsert Yiğit, G.; Kervanci, I.S. Slice-Aware Adversarial Vulnerability Analysis of 
TabNet-Based Intrusion Detection in 5G Network Slicing. Symmetry 2026.
```

---

## License

MIT License — see `LICENSE` for details.
