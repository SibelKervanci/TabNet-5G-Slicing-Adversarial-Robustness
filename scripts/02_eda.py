"""
CICIoT2023 — Keşif Veri Analizi (EDA)
Klasör yapısı: CICIOT2023/train/train.csv
               CICIOT2023/test/test.csv
               CICIOT2023/validation/validation.csv
Çalıştırma  : python3 02_eda.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

# ── 1. YOLLAR ────────────────────────────────────────────────────────────────
BASE = "CICIOT2023"
NROWS = {"train": 200000, "test": 50000, "validation": 50000}
PATHS = {
    "train":      os.path.join(BASE, "train",      "train.csv"),
    "test":       os.path.join(BASE, "test",        "test.csv"),
    "validation": os.path.join(BASE, "validation",  "validation.csv"),
}

for split, path in PATHS.items():
    if not os.path.exists(path):
        print(f"HATA: Bulunamadı: {path}")
        print("   Beklenen yapı: CICIOT2023/train/train.csv")
        exit(1)

# ── 2. YÜKLEME ───────────────────────────────────────────────────────────────
print("Dosyalar yukleniyor...")
splits = {}
for name, path in PATHS.items():
    df = pd.read_csv(path, low_memory=False, nrows=NROWS[name])
    splits[name] = df
    print(f"  {name:12s} -> {len(df):>10,} satir | {df.shape[1]} sutun")

df_all = pd.concat(splits.values(), ignore_index=True)
print(f"\n  TOPLAM    -> {len(df_all):>10,} satir\n")

# ── 3. ETİKET SÜTUNU ─────────────────────────────────────────────────────────
label_candidates = ["label", "Label", "attack_type", "class", "Class", "target", "Attack_type"]
label_col = None
for c in label_candidates:
    if c in df_all.columns:
        label_col = c
        break

if label_col is None:
    print("Tum sutunlar:", df_all.columns.tolist())
    label_col = input("Etiket sutunu adini girin: ").strip()

print(f"Etiket sutunu: '{label_col}'")

# ── 4. SINIF DAĞILIMI ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SINIF DAGILIMI")
print("="*55)

class_counts = df_all[label_col].value_counts()
print(f"\nToplam sinif: {len(class_counts)}")

for split_name, df in splits.items():
    cc = df[label_col].value_counts()
    print(f"\n[{split_name.upper()}] {len(df):,} satir — {len(cc)} sinif")
    for cls, cnt in cc.items():
        print(f"  {str(cls):<35} {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(14, max(6, len(class_counts)*0.45)))
colors = ["#2ECC71" if any(x in str(c).lower() for x in ["benign","normal"])
          else "#E74C3C" for c in class_counts.index]
ax.barh(class_counts.index.astype(str), class_counts.values,
        color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Ornek Sayisi")
ax.set_title("CICIoT2023 — Sinif Dagilimi (Tum Splitler)", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else str(int(x))))
green_p = mpatches.Patch(color="#2ECC71", label="Benign/Normal")
red_p   = mpatches.Patch(color="#E74C3C", label="Saldiri")
ax.legend(handles=[green_p, red_p])
plt.tight_layout()
plt.savefig("reports/01_sinif_dagilimi.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/01_sinif_dagilimi.png")

# ── 5. SPLIT TUTARLILIK KONTROLÜ ─────────────────────────────────────────────
print("\n" + "="*55)
print("SPLIT TUTARLILIK KONTROLU")
print("="*55)
train_cls = set(splits["train"][label_col].unique())
test_cls  = set(splits["test"][label_col].unique())
val_cls   = set(splits["validation"][label_col].unique())

print(f"\n  Train sinif sayisi      : {len(train_cls)}")
print(f"  Test  sinif sayisi      : {len(test_cls)}")
print(f"  Validation sinif sayisi : {len(val_cls)}")

missing_from_test = train_cls - test_cls
missing_from_val  = train_cls - val_cls
extra_in_test     = test_cls  - train_cls

if missing_from_test:
    print(f"\n  UYARI Train'de olup test'te olmayan: {missing_from_test}")
if extra_in_test:
    print(f"  UYARI Test'te olup train'de olmayan: {extra_in_test}")
if not missing_from_test and not extra_in_test:
    print("\n  OK: Tum splitler tutarli — ayni siniflar mevcut")

# ── 6. ÖZELLİK KALİTESİ ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("OZELLIK KALITESI")
print("="*55)

feature_cols = [c for c in df_all.columns
                if c != label_col and df_all[c].dtype in [np.float64, np.int64, float, int]]
print(f"\n  Sayisal ozellik sayisi: {len(feature_cols)}")
print(f"  Toplam sutun          : {df_all.shape[1]}")

df_num = df_all[feature_cols].replace([np.inf, -np.inf], np.nan)
missing = df_num.isnull().sum()
prob    = missing[missing > 0]

if len(prob):
    print(f"\n  UYARI: {len(prob)} sutuunda eksik deger var:")
    for col, cnt in prob.items():
        print(f"    {col:<38} {cnt:,} ({cnt/len(df_all)*100:.1f}%)")
else:
    print("  OK: Eksik deger yok!")

inf_cnt = np.isinf(df_num.fillna(0)).sum()
inf_bad = inf_cnt[inf_cnt > 0]
if len(inf_bad):
    print(f"\n  UYARI: {len(inf_bad)} sutunda sonsuz (inf) deger:")
    print(f"    {inf_bad.index.tolist()}")
else:
    print("  OK: Sonsuz deger yok!")

# ── 7. VARYANS SIRALAMASI ────────────────────────────────────────────────────
print("\n" + "="*55)
print("EN YUKSEK VARYANSLI 20 OZELLIK")
print("(TabNet Attention icin on degerlendirme)")
print("="*55)

feature_var = df_num.var().sort_values(ascending=False)
print()
for i, (feat, var) in enumerate(feature_var.head(20).items(), 1):
    print(f"  {i:2}. {feat:<40} {var:.3e}")

# Korelasyon heatmap
top20 = feature_var.head(20).index.tolist()
fig, ax = plt.subplots(figsize=(14, 12))
corr = df_num[top20].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
            linewidths=0.3, ax=ax, vmin=-1, vmax=1)
ax.set_title("En Yuksek Varyansh 20 Ozellik — Korelasyon\n(TabNet Attention On Analizi)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/02_korelasyon.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/02_korelasyon.png")

# ── 8. ÖZET VE META KAYIT ────────────────────────────────────────────────────
with open("reports/00_ozet.txt", "w", encoding="utf-8") as f:
    f.write("CICIoT2023 EDA OZET\n" + "="*50 + "\n\n")
    f.write(f"Etiket sutunu    : {label_col}\n")
    f.write(f"Toplam ornek     : {len(df_all):,}\n")
    f.write(f"  Train          : {len(splits['train']):,}\n")
    f.write(f"  Test           : {len(splits['test']):,}\n")
    f.write(f"  Validation     : {len(splits['validation']):,}\n")
    f.write(f"Sayisal ozellik  : {len(feature_cols)}\n")
    f.write(f"Sinif sayisi     : {len(class_counts)}\n\n")
    f.write("Sinif Dagilimi:\n")
    for cls, cnt in class_counts.items():
        f.write(f"  {str(cls):<35} {cnt:>10,}\n")
    f.write("\nEn Onemli 15 Ozellik (Varyans):\n")
    for feat, var in feature_var.head(15).items():
        f.write(f"  {feat:<40} {var:.3e}\n")

# Sonraki scriptler bu meta'yı okur
with open("reports/meta.txt", "w") as f:
    f.write(f"label_col={label_col}\n")
    f.write(f"n_features={len(feature_cols)}\n")
    f.write(f"n_classes={len(class_counts)}\n")

print("OK: reports/00_ozet.txt")
print("OK: reports/meta.txt  (sonraki scriptler icin)")

print("\n" + "="*55)
print("EDA TAMAMLANDI")
print("Sonraki adim: python3 03_tabnet_baseline.py")
print("="*55)
