"""
03b_attention_only.py
Kaydedilmiş modeli yükler, sadece attention heatmap üretir.
Çalıştırma: python 03b_attention_only.py
"""

import os, warnings, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OK: PyTorch {torch.__version__} | {DEVICE}")
except ImportError:
    print("HATA: pip install pytorch-tabnet")
    exit(1)

# ── 1. MODELİ YÜKLE ──────────────────────────────────────────────────────────
if not os.path.exists("models/tabnet_ciciot2023.zip"):
    print("HATA: models/tabnet_ciciot2023.zip bulunamadi.")
    exit(1)

model = TabNetClassifier()
model.load_model("models/tabnet_ciciot2023.zip")
print("OK: Model yuklendi")

# ── 2. TEST VERİSİ YÜKLE (küçük örnek) ───────────────────────────────────────
BASE = "CICIOT2023"
print("Test verisi yukleniyor (50K satir)...")
df = pd.read_csv(os.path.join(BASE, "validation", "validation.csv"),
                 low_memory=False, nrows=50000)

label_col = None
for c in ["label","Label","attack_type","class","Class","target","Attack_type"]:
    if c in df.columns:
        label_col = c
        break

feature_cols = [c for c in df.columns
                if c != label_col and df[c].dtype in [np.float64, np.int64, float, int]]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
y_raw = df[label_col].values

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"OK: {len(X_scaled):,} ornek | {len(feature_cols)} ozellik | {len(class_names)} sinif")

# ── 3. ATTENTION ANALİZİ ──────────────────────────────────────────────────────
print("\nAttention analizi basliyor...")

# 2000 örnek yeterli
idx = np.random.choice(len(X_scaled), min(2000, len(X_scaled)), replace=False)
X_sample = X_scaled[idx]

explain_matrix, masks = model.explain(X_sample)

# Genel özellik önemi
feat_imp = pd.Series(
    np.mean(np.abs(explain_matrix), axis=0),
    index=feature_cols
).sort_values(ascending=False)

print("\nEn onemli 15 ozellik (AGE hedefleri):")
for i, (feat, imp) in enumerate(feat_imp.head(15).items(), 1):
    bar = "X" * int(imp / feat_imp.max() * 20)
    print(f"  {i:2}. {feat:<40} {imp:.4f}  {bar}")

# Grafik
fig, ax = plt.subplots(figsize=(12, 7))
top15 = feat_imp.head(15)
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top15)))
ax.barh(range(len(top15)), top15.values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index[::-1], fontsize=9)
ax.set_xlabel("Average Attention Weight")
ax.set_title("")
ax.axvline(x=top15.values.mean(), color="red", linestyle="--", alpha=0.7, label="Mean")
ax.legend()
plt.tight_layout()
plt.savefig("reports/04_attention_ozellik_onemi.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK: reports/04_attention_ozellik_onemi.png")

# ── 4. KARAR ADIMI HEATMAP (hata düzeltildi) ─────────────────────────────────
print(f"\nKarar adimi analizi ({len(masks)} adim)...")

step_imp = []
step_labels = []

for si, mask in enumerate(masks):
    # Farklı mask formatlarını güvenli işle
    try:
        mask_arr = np.array(mask)
        if mask_arr.ndim == 0 or mask_arr.size == 0:
            print(f"  Adim {si+1}: bos, atlaniyor")
            continue
        if mask_arr.ndim == 1:
            imp = np.abs(mask_arr)
        else:
            imp = np.mean(np.abs(mask_arr), axis=0)

        if len(imp) != len(feature_cols):
            print(f"  Adim {si+1}: boyut uyumsuz ({len(imp)} vs {len(feature_cols)}), atlaniyor")
            continue

        step_imp.append(imp)
        step_labels.append(f"Adim {si+1}")
        top3 = pd.Series(imp, index=feature_cols).nlargest(3)
        vals = ", ".join([f"{f}({v:.3f})" for f, v in top3.items()])
        print(f"  Adim {si+1}: {vals}")

    except Exception as e:
        print(f"  Adim {si+1}: hata — {e}, atlaniyor")
        continue

if len(step_imp) > 0:
    top20_idx = feat_imp.head(20).index.tolist()
    top20_pos = [feature_cols.index(f) for f in top20_idx if f in feature_cols]
    step_matrix = np.array(step_imp)[:, top20_pos]

    fig, ax = plt.subplots(figsize=(14, max(3, len(step_imp) * 0.8)))
    sns.heatmap(step_matrix, cmap="YlOrRd", annot=True, fmt=".3f",
                xticklabels=top20_idx,
                yticklabels=step_labels,
                ax=ax, linewidths=0.5)
    ax.set_title("TabNet Karar Adimi x Ozellik Attention Haritasi\n"
                 "(AGE: Hangi Adimda Hangi Ozelligi Hedef Al?)",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/05_adim_attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("OK: reports/05_adim_attention_heatmap.png")
else:
    print("UYARI: Hicbir adim islenemedı, heatmap atlandı")

# ── 5. TOP 5 KAYDET ───────────────────────────────────────────────────────────
feat_imp.head(5).index.to_series().to_csv(
    "reports/top_attention_features.csv", index=False, header=False)
feat_imp.to_csv("reports/full_feature_importance.csv")
print("OK: reports/top_attention_features.csv")

print("\n" + "="*55)
print("ATTENTION ANALİZİ TAMAMLANDI")
print("Sonraki adim: python 04_age_attack.py")
print("="*55)
