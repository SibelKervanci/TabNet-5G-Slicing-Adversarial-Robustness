# Adversarial Vulnerability Analysis of TabNet-Based Intrusion Detection Systems in 5G Network Slicing

Code accompanying the paper:

> Gözde Özsert Yiğit, Ilkay Sibel Kervanci, *"Adversarial Vulnerability
> Analysis of TabNet-Based Intrusion Detection Systems in 5G Network
> Slicing"*, Department of Computer Engineering, Gaziantep University.

This repository contains the full experimental pipeline used in the paper:
TabNet-based NIDS training, attention-mask analysis, the proposed
**Attention-Guided Evasion (AGE)** attack (random, gradient, and
transfer-based variants), the proposed **Per-Slice Robustness Index (PSRI)**
and **Slice-Weighted F1 (SW-F1)** metrics, baseline model comparisons, and
the scripts used to generate every figure in the paper.

## Repository structure

```
.
├── scripts/                     # Main experimental pipeline (run in order)
│   ├── 02_eda.py                 # Exploratory data analysis
│   ├── 03_tabnet_baseline.py     # TabNet training + attention analysis
│   ├── 03b_attention_only.py     # Re-run attention analysis on a saved model
│   ├── 04_age_attack.py          # AGE (random-noise variant) vs FGSM/PGD
│   ├── 04b_age_gradient.py       # AGE gradient-based variant (direct TabNet grad)
│   ├── 04c_age_transfer.py       # AGE transfer-based variant (proposed, paper-final)
│   ├── 07_psri_metric.py         # PSRI / SW-F1 with random-noise AGE
│   ├── 07b_psri_transfer.py      # PSRI / SW-F1 with transfer-based AGE (paper-final)
│   └── 08_baseline_comparison.py # Random Forest / MLP / TabNet comparison
├── figures/                     # Standalone scripts that regenerate paper figures
│   ├── fig_confusion_matrix.py
│   ├── fig1_perslice.py
│   ├── fig2_slice_dagilim.py
│   └── fig3_attack_flow.py
├── data/
│   └── README.md                # Instructions for obtaining CICIoT2023
├── requirements.txt
├── LICENSE
└── .gitignore
```

Running the scripts creates two additional (git-ignored) folders at the
repository root:

- `models/` — saved TabNet checkpoints (`tabnet_ciciot2023.zip`)
- `reports/` — all generated figures (`.png`) and result tables (`.csv`)

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then download the dataset following `data/README.md` and place it at
`CICIOT2023/` in the repository root.

## Reproducing the paper's results

Run from the repository root, in this order:

```bash
# 1. Exploratory data analysis
python scripts/02_eda.py

# 2. Train TabNet baseline + attention-mask analysis
python scripts/03_tabnet_baseline.py

# 3. AGE attack vs FGSM/PGD (transfer-based variant used in the paper)
python scripts/04c_age_transfer.py

# 4. Per-slice robustness metrics (PSRI, SW-F1) — paper-final version
python scripts/07b_psri_transfer.py

# 5. Baseline model comparison (Random Forest / MLP / TabNet)
python scripts/08_baseline_comparison.py

# 6. Regenerate individual paper figures
python figures/fig_confusion_matrix.py
python figures/fig1_perslice.py
python figures/fig2_slice_dagilim.py
python figures/fig3_attack_flow.py
```

`scripts/04_age_attack.py` / `04b_age_gradient.py` and
`scripts/07_psri_metric.py` are earlier/alternative attack formulations
(random-noise and direct-gradient variants) kept for transparency and
ablation purposes; the **transfer-based** versions (`04c_age_transfer.py`,
`07b_psri_transfer.py`) are the ones reported as the main results in the
paper (Sections 3.7, 5.3–5.4).

## Key results

| Metric | Value |
|---|---|
| TabNet baseline accuracy | 97.07% (test), 93.67% (stratified subsample) |
| AGE accuracy drop @ ε=0.10 | 50.2% |
| AGE vs. FGSM perturbation (L2) | 17× less |
| AGE features modified | 3.9 / 46 (8.5%) |
| URLLC PSRI (most vulnerable slice) | 0.023 |
| mMTC PSRI (most robust slice) | 0.983 |
| Global weighted F1 vs. SW-F1 gap | 0.928 → 0.807 (0.121) |

## Citation

If you use this code, please cite:

```bibtex
@article{ozsertyigit2026age,
  title   = {Adversarial Vulnerability Analysis of TabNet-Based Intrusion
             Detection Systems in 5G Network Slicing},
  author  = {\"{O}zsert Yi\u{g}it, G\"{o}zde and Kervanci, Ilkay Sibel},
  journal = {},
  year    = {2026}
}
```

## License

This project is released under the MIT License — see [LICENSE](LICENSE).
The CICIoT2023 dataset itself is subject to its own license/terms from the
Canadian Institute for Cybersecurity; see `data/README.md`.
