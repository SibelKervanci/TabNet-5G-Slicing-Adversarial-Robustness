"""
fig1_perslice.py
Makale Fig.1 — Per-Slice F1 vs Global F1 grafiği
Çalıştırma: python fig1_perslice.py
Çıktı: reports/fig1_perslice_f1.png (300 DPI, makaleye hazır)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("reports", exist_ok=True)

# ── VERİ — PSRI scriptinden alınan gerçek sonuçlar ───────────────────────────
categories = ["Global\n(Weighted)", "eMBB", "URLLC", "mMTC", "Benign"]
f1_clean   = [0.928, 0.981, 0.956, 0.483, 0.982]
f1_fgsm    = [0.521, 0.615, 0.549, 0.476, 0.957]

x     = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

# Barlar
bars1 = ax.bar(x - width/2, f1_clean, width,
               label="Clean (No Attack)", color="#2E5090",
               alpha=0.88, edgecolor="white", linewidth=1.2)
bars2 = ax.bar(x + width/2, f1_fgsm, width,
               label="Under FGSM Attack", color="#C0392B",
               alpha=0.88, edgecolor="white", linewidth=1.2)

# Değer etiketleri
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#2E5090")

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#C0392B")

# mMTC annotation — makalenin ana mesajı
ax.annotate(
    "Global F1 hides\nmMTC collapse\n(0.928 → 0.483)",
    xy=(x[3] - width/2, 0.483),
    xytext=(x[3] + 0.9, 0.62),
    fontsize=9, color="#7B4F00", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#7B4F00", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDF6E3",
              edgecolor="#7B4F00", alpha=0.9)
)

# Global F1 referans çizgisi
ax.axhline(y=0.928, color="#2E5090", linestyle="--",
           linewidth=1.2, alpha=0.5)
ax.text(4.6, 0.933, "Global F1\n= 0.928",
        fontsize=8.5, color="#2E5090", alpha=0.8, ha="right")

# Eksen ayarları
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
ax.set_ylabel("Weighted F1-Score", fontsize=12)
ax.set_ylim(0, 1.10)
ax.set_title("")

# Dilim arka plan bantları
ax.axvspan(-0.5, 0.5, alpha=0.04, color="gray")
ax.axvspan( 0.5, 1.5, alpha=0.06, color="#2E5090")
ax.axvspan( 1.5, 2.5, alpha=0.06, color="#E67E22")
ax.axvspan( 2.5, 3.5, alpha=0.08, color="#C0392B")
ax.axvspan( 3.5, 4.5, alpha=0.04, color="#27AE60")

# Dilim etiketleri (alt)
for xi, (lbl, col) in enumerate(zip(
    ["—", "eMBB", "URLLC", "mMTC", "Benign"],
    ["gray", "#2E5090", "#E67E22", "#C0392B", "#27AE60"]
)):
    ax.text(xi, -0.09, lbl, ha="center", fontsize=8.5,
            color=col, fontweight="bold",
            transform=ax.get_xaxis_transform())

ax.legend(fontsize=10, loc="upper left",
          framealpha=0.9, edgecolor="#CCCCCC")
ax.grid(True, axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("reports/fig1_perslice_f1.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("OK: reports/fig1_perslice_f1.png")
print("Makaleye eklemek icin bu dosyayi kullanin.")
