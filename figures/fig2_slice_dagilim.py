"""
fig2_slice_dagilim.py
Makale Fig.2 — 5G Dilim Bazlı Örnek Dağılımı (Pie Chart)
Çalıştırma: python fig2_slice_dagilim.py
Çıktı: reports/fig2_slice_dagilim.png (300 DPI)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("reports", exist_ok=True)

# ── VERİ — EDA çıktısından alınan gerçek sayılar ─────────────────────────────
dilimler    = ["eMBB", "URLLC", "mMTC", "Benign"]
ornekler    = [141167, 36605, 18313, 4572]   # train seti
yüzdeler    = [70.5, 18.3, 6.6, 2.3]        # yaklaşık
saldilar    = [12, 6, 15, 0]                 # saldırı türü sayısı
renkler     = ["#2E5090", "#E67E22", "#C0392B", "#27AE60"]
patlama     = [0.04, 0.04, 0.12, 0.04]      # mMTC patlatılmış — dikkat çeksin

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor("white")

# ── SOL: Örnek sayısı pie ─────────────────────────────────────────────────────
ax1 = axes[0]
wedges, texts, autotexts = ax1.pie(
    ornekler,
    labels=None,
    colors=renkler,
    explode=patlama,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.75,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)

for at, renk in zip(autotexts, renkler):
    at.set_fontsize(11)
    at.set_fontweight("bold")
    at.set_color("white")

# Manuel legend
legend_labels = [
    f"eMBB  — {ornekler[0]:,} samples ({yüzdeler[0]}%)\n  12 attack types",
    f"URLLC — {ornekler[1]:,} samples ({yüzdeler[1]}%)\n  6 attack types",
    f"mMTC  — {ornekler[2]:,} samples ({yüzdeler[2]}%)\n  15 attack types",
    f"Benign — {ornekler[3]:,} samples ({yüzdeler[3]}%)",
]
patches = [mpatches.Patch(color=r, label=l) for r, l in zip(renkler, legend_labels)]
ax1.legend(handles=patches, loc="lower left", fontsize=8.5,
           framealpha=0.9, edgecolor="#CCCCCC",
           bbox_to_anchor=(-0.15, -0.18))
ax1.set_title("(a) Distribution by Slice",
              fontsize=11, fontweight="bold", pad=16)

# mMTC annotation
ax1.annotate(
    "mMTC: only\n6.6% of samples\nbut 15 attack types",
    xy=(-0.35, -0.85), xytext=(-0.9, -0.6),
    fontsize=8.5, color="#6B1111", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#6B1111", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDE8E8",
              edgecolor="#6B1111", alpha=0.9)
)

# ── SAĞ: Saldırı türü sayısı bar ─────────────────────────────────────────────
ax2 = axes[1]
x = np.arange(len(dilimler[:3]))  # Benign hariç
bar_renkler = renkler[:3]
bar_vals = saldilar[:3]

bars = ax2.bar(x, bar_vals, color=bar_renkler, alpha=0.88,
               edgecolor="white", linewidth=1.5, width=0.5)

for bar, val in zip(bars, bar_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(val), ha="center", va="bottom",
             fontsize=13, fontweight="bold", color=bar.get_facecolor())

ax2.set_xticks(x)
ax2.set_xticklabels(["eMBB", "URLLC", "mMTC"], fontsize=12, fontweight="bold")
ax2.set_ylabel("Number of Attack Types", fontsize=11)
ax2.set_ylim(0, 20)
ax2.set_title("(b) Number of Attack Types per Slice",
              fontsize=11, fontweight="bold", pad=16)
ax2.grid(True, axis="y", alpha=0.3, linestyle="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# mMTC vurgu kutusu
ax2.annotate(
    "Fewest samples\nmost attack types\n→ Most vulnerable",
    xy=(2, 15), xytext=(1.3, 18),
    fontsize=8.5, color="#6B1111", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#6B1111", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDE8E8",
              edgecolor="#6B1111", alpha=0.9)
)


plt.tight_layout()
plt.savefig("reports/fig2_slice_dagilim.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("OK: reports/fig2_slice_dagilim.png")
