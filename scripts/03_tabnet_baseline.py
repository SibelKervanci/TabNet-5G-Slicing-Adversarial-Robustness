"""
CICIoT2023 — TabNet Baseline + Attention Mask Analizi
Klasör yapısı: CICIOT2023/train/train.csv  vb.
Çalıştırma  : python3 03_tabnet_baseline.py
Gereksinim  : pip install pytorch-tabnet scikit-learn
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OK: PyTorch {torch.__version__} | Cihaz: {DEVICE}")
except ImportError:
    print("HATA: pip install pytorch-tabnet")
    exit(1)

# ── 1. VERİ YÜKLE ────────────────────────────────────────────────────────────
BASE = "CICIOT2023"

def load(split):
    path = os.path.join(BASE, split, f"{split}.csv")
    return pd.read_csv(path, low_memory=False)

print("\nVeri yukleniyor...")
df_train = load("train")
df_test  = load("test")
df_val   = load("validation")
print(f"  Train: {len(df_train):,} | Test: {len(df_test):,} | Val: {len(df_val):,}")

# ── 2. ETİKET SÜTUNU ─────────────────────────────────────────────────────────
meta_label = None
if os.path.exists("reports/meta.txt"):
    for line in open("reports/meta.txt"):
        if line.startswith("label_col="):
            meta_label = line.strip().split("=")[1]

if meta_label and meta_label in df_train.columns:
    label_col = meta_label
else:
    for c in ["label","Label","attack_type","class","Class","target","Attack_type"]:
        if c in df_train.columns:
            label_col = c
            break

print(f"  Etiket sutunu: '{label_col}'")

# ── 3. ÖN İŞLEME ─────────────────────────────────────────────────────────────
feature_cols = [c for c in df_train.columns
                if c != label_col and df_train[c].dtype in [np.float64, np.int64, float, int]]
print(f"  Ozellik sayisi: {len(feature_cols)}")

def prep(df):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df[label_col].values
    return X, y

X_train_r, y_train_r = prep(df_train)
X_test_r,  y_test_r  = prep(df_test)
X_val_r,   y_val_r   = prep(df_val)

le = LabelEncoder().fit(np.concatenate([y_train_r, y_test_r, y_val_r]))
y_train = le.transform(y_train_r)
y_test  = le.transform(y_test_r)
y_val   = le.transform(y_val_r)
class_names = le.classes_
print(f"  Sinif sayisi: {len(class_names)}")
print(f"  Siniflar: {list(class_names)}")

scaler = StandardScaler().fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test  = scaler.transform(X_test_r)
X_val   = scaler.transform(X_val_r)

# ── 4. TABNET EĞİTİMİ ────────────────────────────────────────────────────────
print("\nTabNet egitimi basliyor...")
print("  n_steps=5 → 5G slice analizi icin 5 karar adimi")

model = TabNetClassifier(
    n_d=32, n_a=32,
    n_steps=5,      # Her adım farklı özelliklere dikkat eder
    gamma=1.3,
    n_independent=2, n_shared=2,
    momentum=0.02,
    seed=42,
    device_name=DEVICE,
    verbose=1
)

model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_val, y_val)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=50,
    patience=10,
    batch_size=4096,
    virtual_batch_size=256,
    num_workers=0,
    drop_last=False
)

model.save_model("models/tabnet_ciciot2023")
print("OK: models/tabnet_ciciot2023.zip")

# ── 5. PERFORMANS ────────────────────────────────────────────────────────────
print("\nTest performansi...")
y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"  Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
print()
print(classification_report(y_test, y_pred, target_names=class_names))

fig, ax = plt.subplots(figsize=(28, 24))  # daha b�y�k
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("Actual Label", fontsize=13)
ax.set_title("")
plt.xticks(rotation=90, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("reports/03_confusion_matrix.png",
            dpi=200, bbox_inches="tight", facecolor="white")

# ── 6. ATTENTION MASK ANALİZİ ────────────────────────────────────────────────
print("\nAttention mask analizi (AGE icin kritik)...")

# Test setinden 2000 örnek al (hız için)
idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
X_sample = X_test[idx]

explain_matrix, masks = model.explain(X_sample)

# Genel özellik önemi
feat_imp = pd.Series(
    np.mean(np.abs(explain_matrix), axis=0),
    index=feature_cols
).sort_values(ascending=False)

print("\nEn onemli 15 ozellik (AGE saldirisi icin hedefler):")
for i, (feat, imp) in enumerate(feat_imp.head(15).items(), 1):
    bar = "X" * int(imp / feat_imp.max() * 20)
    print(f"  {i:2}. {feat:<40} {imp:.4f}  {bar}")

# Grafik: özellik önemi
fig, ax = plt.subplots(figsize=(12, 7))
top15 = feat_imp.head(15)
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top15)))
ax.barh(range(len(top15)), top15.values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index[::-1], fontsize=9)
ax.set_xlabel("Ortalama Attention Agirligi")
ax.set_title("TabNet Attention Mask — En Onemli 15 Ozellik\n(AGE Saldirisi Icin Birincil Hedefler)",
             fontsize=12, fontweight="bold")
ax.axvline(x=top15.values.mean(), color="red", linestyle="--", alpha=0.7, label="Ortalama")
ax.legend()
plt.tight_layout()
plt.savefig("reports/04_attention_ozellik_onemi.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK: reports/04_attention_ozellik_onemi.png")

# Karar adımı × özellik heatmap
print(f"\nKarar adimi bazinda attention ({len(masks)} adim):")
step_imp = []
for si, mask in enumerate(masks):
    mask_arr = mask if isinstance(mask, np.ndarray) else np.array(mask)
    if mask_arr.ndim == 0 or mask_arr.size == 0:
        continue
    imp = np.mean(np.abs(mask_arr), axis=0) if mask_arr.ndim > 1 else np.abs(mask_arr)
    step_imp.append(imp)
    top3 = pd.Series(imp, index=feature_cols).nlargest(3)
    vals = ", ".join([f"{f}({v:.3f})" for f,v in top3.items()])
    print(f"  Adim {si+1}: {vals}")

top20_idx = feat_imp.head(20).index.tolist()
top20_pos = [feature_cols.index(f) for f in top20_idx if f in feature_cols]
step_matrix = np.array(step_imp)[:, top20_pos]

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(step_matrix, cmap="YlOrRd", annot=True, fmt=".3f",
            xticklabels=top20_idx,
            yticklabels=[f"Adim {i+1}" for i in range(len(masks))],
            ax=ax, linewidths=0.5)
ax.set_title("TabNet Karar Adimi x Ozellik Attention Haritasi\n"
             "(AGE: Hangi Adimda Hangi Ozelligi Hedef Al?)",
             fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("reports/05_adim_attention_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK: reports/05_adim_attention_heatmap.png")

# Top 5'i kaydet
feat_imp.head(5).index.to_series().to_csv(
    "reports/top_attention_features.csv", index=False, header=False)
feat_imp.to_csv("reports/full_feature_importance.csv")
print("OK: reports/top_attention_features.csv  (AGE icin)")

print("\n" + "="*55)
print("BASELINE EGITIMI TAMAMLANDI")
print(f"Baseline accuracy: {acc:.4f}")
print("Sonraki adim: python3 04_age_attack.py")
print("="*55)
