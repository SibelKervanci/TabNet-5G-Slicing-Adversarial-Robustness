"""
fig3_attack_flow.py
Fig.3 — Attack Flow Diagram (large fonts, high resolution)
Output: reports/fig3_attack_flow.png (300 DPI)
Run   : python fig3_attack_flow.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import os

os.makedirs("reports", exist_ok=True)

# ── CANVAS ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(28, 18))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 28)
ax.set_ylim(0, 18)
ax.axis("off")

# ── RENKLER ──────────────────────────────────────────────────────────────────
C = {
    "blue":    "#2E5090", "lblue":   "#D6E4F7",
    "purple":  "#534AB7", "lpurple": "#EEEDFE",
    "teal":    "#0F6E56", "lteal":   "#D6F0E8",
    "coral":   "#993C1D", "lcoral":  "#FAECE7",
    "amber":   "#BA7517", "lamber":  "#FDF0D5",
    "green":   "#3B6D11", "lgreen":  "#E5F0D8",
    "gray":    "#4A4A4A", "lgray":   "#F2F4F7",
}

# ── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────────
def lane(x, y, w, h, title, fc="#F8F8F8"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.2",
                          facecolor=fc, edgecolor="#BBBBBB",
                          linewidth=1.2, zorder=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.6, title,
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333", zorder=2)

def box(x, y, w, h, label, sublabel=None, fc="#EEF3FA", ec="#2E5090", tc="#2E5090"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=fc, edgecolor=ec,
                          linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h * 0.65, label,
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=tc, zorder=4)
        ax.text(x + w/2, y + h * 0.28, sublabel,
                ha="center", va="center",
                fontsize=10.5, color=tc, alpha=0.88, zorder=4)
    else:
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=tc, zorder=4)

def step(x, y, n, ec="#888888"):
    circle = plt.Circle((x, y), 0.38,
                         color="white", ec=ec, lw=1.2, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, str(n), ha="center", va="center",
            fontsize=11, fontweight="bold", color=ec, zorder=6)

def arrow(x1, y1, x2, y2, color="#555555", lw=1.6, dash=False):
    ls = (0, (5, 3)) if dash else "solid"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=lw,
                                linestyle=ls,
                                mutation_scale=16),
                zorder=2)

def label_text(x, y, text, color="#555555", fs=10):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color, style="italic", zorder=4)

# ── LANE'LER ─────────────────────────────────────────────────────────────────
lane(0.5,  0.5,  8.0, 17.0, "Data & Model",   fc="#EEF4FF")
lane(9.5,  0.5,  9.0, 17.0, "Attack",         fc="#FFF7EE")
lane(19.5, 0.5,  8.0, 17.0, "Evaluation",     fc="#EEFFF5")

BW, BH = 6.6, 1.6

# ── SOL: Data & Model ────────────────────────────────────────────────────────
# 1. Dataset
box(0.9, 14.5, BW, BH,
    "CICIoT2023 Dataset",
    "34 classes  ·  46 features  ·  ~7.8M samples",
    fc=C["lblue"], ec=C["blue"], tc=C["blue"])
step(0.9, 15.3, 1, C["blue"])

# 2. Preprocessing
box(0.9, 12.3, BW, BH,
    "Preprocessing",
    "StandardScaler  ·  No missing/infinite values",
    fc=C["lgray"], ec=C["gray"], tc=C["gray"])
step(0.9, 13.1, 2, C["gray"])

# 3. TabNet Training
box(0.9, 10.1, BW, BH,
    "TabNet Training",
    "Acc: 97.07%  ·  Macro F1: 0.578  ·  n_steps=5",
    fc=C["lpurple"], ec=C["purple"], tc=C["purple"])
step(0.9, 10.9, 3, C["purple"])

# 4. Attention Mask Analysis
box(0.9, 7.9, BW, BH,
    "Attention Mask Analysis",
    "Top-5: syn_count, IAT, SMTP, UDP, TCP",
    fc=C["lteal"], ec=C["teal"], tc=C["teal"])
step(0.9, 8.7, 4, C["teal"])

# 5. Surrogate MLP
box(0.9, 5.7, BW, BH,
    "Surrogate MLP Training",
    "256-128-64 neurons  ·  Transfer for AGE",
    fc=C["lamber"], ec=C["amber"], tc=C["amber"])
step(0.9, 6.5, 5, C["amber"])

# 6. 5G Slice Mapping
box(0.9, 1.8, BW, BH,
    "5G Network Slice Mapping",
    "eMBB  ·  URLLC  ·  mMTC",
    fc=C["lgray"], ec=C["gray"], tc=C["gray"])
step(0.9, 2.6, 6, C["gray"])

# Sol oklar
arrow(4.2, 14.5, 4.2, 14.0, C["blue"])
arrow(4.2, 12.3, 4.2, 11.8, C["gray"])
arrow(4.2, 10.1, 4.2,  9.6, C["purple"])
arrow(4.2,  7.9, 4.2,  7.4, C["teal"])
arrow(4.2,  5.7, 4.2,  3.5, C["amber"], dash=True)

# ── ORTA: Attack ─────────────────────────────────────────────────────────────
AX, AW, AH = 10.0, 8.0, 1.6

# Epsilon kutu
box(10.0, 15.8, AW, 0.9,
    "\u03b5  \u2208  {0.01,  0.05,  0.10,  0.20,  0.30,  0.50}",
    fc="white", ec="#AAAAAA", tc="#555555")

# 7. FGSM
box(AX, 13.0, AW, AH,
    "FGSM",
    "All 46 features  ·  Single-step gradient sign",
    fc=C["lcoral"], ec=C["coral"], tc=C["coral"])
step(AX, 13.8, 7, C["coral"])

# 8. PGD
box(AX, 10.8, AW, AH,
    "PGD",
    "All 46 features  ·  10 iterative steps",
    fc=C["lcoral"], ec=C["coral"], tc=C["coral"])
step(AX, 11.6, 8, C["coral"])

# 9. AGE (Transfer)
box(AX, 8.6, AW, AH,
    "AGE — Transfer-Based (Proposed)",
    "Top-5 attention features  ·  Surrogate \u2192 TabNet",
    fc=C["lamber"], ec=C["amber"], tc=C["amber"])
step(AX, 9.4, 9, C["amber"])

# Model → Attack (yatay)
arrow(7.5, 11.0, 10.0, 11.0, C["purple"])
label_text(8.75, 11.35, "test set", C["purple"], fs=10)

# Attention → AGE
arrow(7.5, 8.7, 10.0, 9.0, C["teal"], dash=True)
label_text(8.75, 9.1, "attention mask", C["teal"], fs=10)

# Surrogate → AGE
arrow(7.5, 6.3, 10.0, 8.8, C["amber"], dash=True)
label_text(8.5, 7.8, "surrogate gradient", C["amber"], fs=10)

# ── SAĞ: Evaluation ──────────────────────────────────────────────────────────
EX, EW, EH = 20.0, 7.0, 1.6

# 10. PSRI Calculation
box(EX, 13.0, EW, EH,
    "PSRI Calculation",
    "Per-Slice Robustness Index  ·  3 slices",
    fc=C["lteal"], ec=C["teal"], tc=C["teal"])
step(EX, 13.8, 10, C["teal"])

# 11. Metric Comparison
box(EX, 10.8, EW, EH,
    "Global vs Slice Metrics",
    "Weighted F1  vs  SW-F1",
    fc=C["lteal"], ec=C["teal"], tc=C["teal"])
step(EX, 11.6, 11, C["teal"])

# 12. Stealthiness Analysis
box(EX, 8.6, EW, EH,
    "Stealthiness Analysis",
    "L\u2082 norm  ·  Features modified",
    fc=C["lteal"], ec=C["teal"], tc=C["teal"])
step(EX, 9.4, 12, C["teal"])

# Attack → Evaluation
arrow(18.0, 13.8, 20.0, 13.8, C["coral"])
arrow(18.0, 11.6, 20.0, 11.6, C["coral"])
arrow(18.0,  9.4, 20.0,  9.4, C["amber"])

# ── RESULTS ──────────────────────────────────────────────────────────────────
box(9.5, 1.8, 18.0, 1.6,
    "Results & Discussion",
    "PSRI  ·  SW-F1  ·  L\u2082 stealthiness  ·  Baseline comparison",
    fc=C["lgreen"], ec=C["green"], tc=C["green"])

# Evaluation → Results
ax.annotate("", xy=(23.5, 3.4), xytext=(23.5, 8.6),
            arrowprops=dict(arrowstyle="-|>", color=C["teal"],
                            lw=1.6, mutation_scale=16), zorder=2)

# Slice Mapping → Results
arrow(4.2, 1.8, 9.5, 2.6, C["gray"], dash=True)

plt.tight_layout(pad=0.2)
plt.savefig("reports/fig3_attack_flow.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("OK: reports/fig3_attack_flow.png")