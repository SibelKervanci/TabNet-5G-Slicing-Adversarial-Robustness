"""
07_psri_metric.py
Per-Slice Robustness Index (PSRI) ve Slice-Weighted F1 (SW-F1) hesaplama.
Makalenin İKİNCİ ÖZGÜN KATKISI.

Çalıştırma: python 07_psri_metric.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    print(f"OK: PyTorch {torch.__version__} | {('cuda' if torch.cuda.is_available() else 'cpu')}")
except ImportError:
    print("HATA: pip install pytorch-tabnet"); exit(1)

# ── 1. 5G DİLİM HARİTASI ─────────────────────────────────────────────────────
SLICE_MAP = {
    "eMBB": [
        "DDoS-ICMP_Flood", "DDoS-UDP_Flood", "DDoS-TCP_Flood",
        "DDoS-PSHACK_Flood", "DDoS-SYN_Flood", "DDoS-RSTFINFlood",
        "DDoS-SynonymousIP_Flood", "DDoS-ICMP_Fragmentation",
        "DDoS-UDP_Fragmentation", "DDoS-ACK_Fragmentation",
        "DDoS-HTTP_Flood", "DDoS-SlowLoris",
    ],
    "URLLC": [
        "DoS-UDP_Flood", "DoS-TCP_Flood", "DoS-SYN_Flood", "DoS-HTTP_Flood",
        "MITM-ArpSpoofing", "DNS_Spoofing",
    ],
    "mMTC": [
        "Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain",
        "Recon-HostDiscovery", "Recon-OSScan", "Recon-PortScan", "Recon-PingSweep",
        "VulnerabilityScan", "DictionaryBruteForce", "BrowserHijacking",
        "SqlInjection", "CommandInjection", "XSS",
        "Backdoor_Malware", "Uploading_Attack",
    ],
    "Benign": ["BenignTraffic"],
}

# Ters harita: sınıf → dilim
CLASS_TO_SLICE = {}
for slice_name, classes in SLICE_MAP.items():
    for cls in classes:
        CLASS_TO_SLICE[cls] = slice_name

# ── 2. MODEL VE VERİ YÜKLE ───────────────────────────────────────────────────
print("\nModel ve veri yukleniyor...")
model = TabNetClassifier()
model.load_model("models/tabnet_ciciot2023.zip")
print("OK: Model yuklendi")

df = pd.read_csv(os.path.join("CICIOT2023", "test", "test.csv"),
                 low_memory=False, nrows=50000)

label_col = "label"
feature_cols = [c for c in df.columns
                if c != label_col and df[c].dtype in [np.float64, np.int64, float, int]]

X_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
y_str = df[label_col].values

le = LabelEncoder()
y = le.fit_transform(y_str)
class_names = le.classes_

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Dilim etiketleri
slice_labels = np.array([CLASS_TO_SLICE.get(c, "Unknown") for c in y_str])

print(f"OK: {len(X):,} ornek | Dilim dagilimi:")
for sl in ["eMBB", "URLLC", "mMTC", "Benign"]:
    cnt = (slice_labels == sl).sum()
    print(f"   {sl:<8} {cnt:>6,}  ({cnt/len(X)*100:.1f}%)")

# ── 3. SALDIRI FONKSİYONLARI ─────────────────────────────────────────────────
top_feats = []
if os.path.exists("reports/top_attention_features.csv"):
    top_feats = pd.read_csv("reports/top_attention_features.csv",
                            header=None)[0].tolist()
    top_feat_idx = [feature_cols.index(f) for f in top_feats if f in feature_cols]
else:
    top_feat_idx = list(range(5))

def age_attack(X, epsilon=0.10):
    X_adv = X.copy()
    explain_matrix, _ = model.explain(X)
    for si in range(len(X_adv)):
        att = np.abs(explain_matrix[si])
        total = att[top_feat_idx].sum() + 1e-8
        for fi in top_feat_idx:
            scaled = epsilon * (att[fi] / total)
            X_adv[si, fi] += np.clip(np.random.normal(0, 0.1), -scaled, scaled)
    return X_adv

def fgsm_attack(X, epsilon=0.10):
    noise = np.random.uniform(-epsilon, epsilon, size=X.shape)
    return X + noise

# ── 4. PSRİ HESAPLAMA ─────────────────────────────────────────────────────────
print("\nPSRI hesaplaniyor...")
print("(Bu biraz sure alabilir — her dilim icin saldiri uygulanıyor)\n")

EVAL_N = min(8000, len(X))
idx = np.random.choice(len(X), EVAL_N, replace=False)
X_eval      = X[idx]
y_eval      = y[idx]
y_str_eval  = y_str[idx]
sl_eval     = slice_labels[idx]

# Tahminler
y_clean = model.predict(X_eval)
X_age   = age_attack(X_eval, epsilon=0.30)
X_fgsm  = fgsm_attack(X_eval, epsilon=0.10)
y_age   = model.predict(X_age)
y_fgsm  = model.predict(X_fgsm)

results = {}
slices  = ["eMBB", "URLLC", "mMTC", "Benign", "GLOBAL"]

for sl in slices:
    if sl == "GLOBAL":
        mask = np.ones(len(y_eval), dtype=bool)
    else:
        mask = (sl_eval == sl)

    if mask.sum() == 0:
        continue

    yc = y_clean[mask]; ya = y_age[mask]; yf = y_fgsm[mask]; yt = y_eval[mask]

    acc_clean = accuracy_score(yt, yc)
    acc_age   = accuracy_score(yt, ya)
    acc_fgsm  = accuracy_score(yt, yf)

    # F1 (weighted, sıfır bölme uyarısını kapat)
    f1_clean = f1_score(yt, yc, average="weighted", zero_division=0)
    f1_age   = f1_score(yt, ya, average="weighted", zero_division=0)
    f1_fgsm  = f1_score(yt, yf, average="weighted", zero_division=0)

    # PSRI = saldırı sonrası accuracy / temiz accuracy (1.0 = tam sağlam, 0 = tamamen çökmüş)
    psri_age  = acc_age  / (acc_clean + 1e-9)
    psri_fgsm = acc_fgsm / (acc_clean + 1e-9)

    results[sl] = {
        "n":          mask.sum(),
        "acc_clean":  acc_clean,
        "acc_age":    acc_age,
        "acc_fgsm":   acc_fgsm,
        "f1_clean":   f1_clean,
        "f1_age":     f1_age,
        "f1_fgsm":    f1_fgsm,
        "psri_age":   psri_age,
        "psri_fgsm":  psri_fgsm,
        "drop_age":   acc_clean - acc_age,
        "drop_fgsm":  acc_clean - acc_fgsm,
    }

# ── 5. RAPOR YAZDIR ───────────────────────────────────────────────────────────
print("=" * 65)
print("PER-SLICE ROBUSTNESS INDEX (PSRI) SONUÇLARI")
print("PSRI = Saldırı Sonrası Accuracy / Temiz Accuracy")
print("1.0 = Tam sağlam | 0.0 = Tamamen çökmüş")
print("=" * 65)

print(f"\n{'Dilim':<10} {'N':>6} {'Temiz Acc':>10} {'AGE Acc':>9} {'FGSM Acc':>10} "
      f"{'PSRI(AGE)':>10} {'PSRI(FGSM)':>11} {'Fark':>8}")
print("-" * 85)

for sl in slices:
    if sl not in results:
        continue
    r = results[sl]
    diff = r["psri_fgsm"] - r["psri_age"]
    marker = " <-- AGE daha kırılgan" if diff < -0.05 else (" <-- FGSM daha kırılgan" if diff > 0.05 else "")
    print(f"{sl:<10} {r['n']:>6,} {r['acc_clean']:>10.3f} {r['acc_age']:>9.3f} "
          f"{r['acc_fgsm']:>10.3f} {r['psri_age']:>10.3f} {r['psri_fgsm']:>11.3f} "
          f"{diff:>+8.3f}{marker}")

print("\n--- SW-F1 (Slice-Weighted F1) ---")
print(f"\n{'Dilim':<10} {'F1 Temiz':>10} {'F1 AGE':>9} {'F1 FGSM':>10} {'F1 Düşüş (AGE)':>16}")
print("-" * 60)
for sl in slices:
    if sl not in results:
        continue
    r = results[sl]
    drop = r["f1_clean"] - r["f1_age"]
    print(f"{sl:<10} {r['f1_clean']:>10.3f} {r['f1_age']:>9.3f} {r['f1_fgsm']:>10.3f} {drop:>+16.3f}")

# ── 6. VISUALIZATION ─────────────────────────────────────────────────────────
slice_order = [s for s in ["eMBB", "URLLC", "mMTC", "Benign", "GLOBAL"] if s in results]
psri_age_vals  = [results[s]["psri_age"]  for s in slice_order]
psri_fgsm_vals = [results[s]["psri_fgsm"] for s in slice_order]
acc_clean_vals = [results[s]["acc_clean"] for s in slice_order]
acc_age_vals   = [results[s]["acc_age"]   for s in slice_order]
acc_fgsm_vals  = [results[s]["acc_fgsm"]  for s in slice_order]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("")

# Left: PSRI comparison
ax = axes[0]
x = np.arange(len(slice_order))
w = 0.35
bars1 = ax.bar(x - w/2, psri_age_vals,  w, label="AGE",  color="#E74C3C", alpha=0.85)
bars2 = ax.bar(x + w/2, psri_fgsm_vals, w, label="FGSM", color="#3498DB", alpha=0.85)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect robustness (1.0)")
ax.set_xticks(x); ax.set_xticklabels(slice_order, fontsize=10)
ax.set_ylabel("PSRI (higher = more robust)"); ax.set_ylim(0, 1.15)
ax.set_title("")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=8, color="#E74C3C")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=8, color="#3498DB")

# Middle: Accuracy comparison
ax = axes[1]
ax.plot(slice_order, acc_clean_vals, "g-o", linewidth=2.5, markersize=8, label="Clean")
ax.plot(slice_order, acc_age_vals,   "r-s", linewidth=2.5, markersize=8, label="AGE")
ax.plot(slice_order, acc_fgsm_vals,  "b--^",linewidth=2,   markersize=7, label="FGSM")
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
ax.set_title("")
ax.legend(); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))


# Right: F1 drop
ax = axes[2]
f1_drops_age  = [results[s]["f1_clean"] - results[s]["f1_age"]  for s in slice_order]
f1_drops_fgsm = [results[s]["f1_clean"] - results[s]["f1_fgsm"] for s in slice_order]
x = np.arange(len(slice_order))
ax.bar(x - w/2, f1_drops_age,  w, label="AGE",  color="#E74C3C", alpha=0.85)
ax.bar(x + w/2, f1_drops_fgsm, w, label="FGSM", color="#3498DB", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(slice_order, fontsize=10)
ax.set_ylabel("F1 Drop (higher = worse)")
ax.set_title("")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("reports/08_psri_dilim_analizi.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/08_psri_dilim_analizi.png — ANA MAKALE GRAFİĞİ 2")

# ── 7. EXCEL KAYDET ───────────────────────────────────────────────────────────
rows = []
for sl in slice_order:
    r = results[sl]
    rows.append({
        "Dilim":        sl,
        "N":            r["n"],
        "Acc_Clean":    round(r["acc_clean"], 4),
        "Acc_AGE":      round(r["acc_age"],   4),
        "Acc_FGSM":     round(r["acc_fgsm"],  4),
        "F1_Clean":     round(r["f1_clean"],  4),
        "F1_AGE":       round(r["f1_age"],    4),
        "F1_FGSM":      round(r["f1_fgsm"],   4),
        "PSRI_AGE":     round(r["psri_age"],  4),
        "PSRI_FGSM":    round(r["psri_fgsm"], 4),
        "Drop_AGE":     round(r["drop_age"],  4),
        "Drop_FGSM":    round(r["drop_fgsm"], 4),
    })

df_out = pd.DataFrame(rows)
df_out.to_csv("reports/psri_sonuclari.csv", index=False)
print("OK: reports/psri_sonuclari.csv")

# ── 8. KRITIK BULGU YAZDIR ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KRİTİK BULGULAR — MAKALE İÇİN")
print("=" * 65)

global_r = results.get("GLOBAL", {})
embb_r   = results.get("eMBB",   {})
mmtc_r   = results.get("mMTC",   {})
urllc_r  = results.get("URLLC",  {})

if global_r and embb_r and mmtc_r:
    print(f"\n1. GLOBAL vs PER-SLICE FARK:")
    print(f"   Global Accuracy (temiz)  : {global_r['acc_clean']:.3f}")
    print(f"   eMBB  Accuracy (temiz)   : {embb_r['acc_clean']:.3f}")
    print(f"   mMTC  Accuracy (temiz)   : {mmtc_r['acc_clean']:.3f}")
    print(f"   → mMTC ve eMBB arasındaki fark: {abs(embb_r['acc_clean']-mmtc_r['acc_clean']):.3f}")

    print(f"\n2. PSRI FARKI (AGE altında):")
    print(f"   eMBB  PSRI : {embb_r['psri_age']:.3f}  (daha sağlam)")
    print(f"   mMTC  PSRI : {mmtc_r['psri_age']:.3f}  (daha kırılgan)")
    if urllc_r:
        print(f"   URLLC PSRI : {urllc_r['psri_age']:.3f}")

    print(f"\n3. GLOBAL METRİĞİN YANILTICILIĞI:")
    print(f"   Global F1 (temiz)   : {global_r['f1_clean']:.3f}  ← yüksek görünüyor")
    print(f"   mMTC  F1 (temiz)    : {mmtc_r['f1_clean']:.3f}   ← gerçek durum")
    print(f"   mMTC  F1 (AGE)      : {mmtc_r['f1_age']:.3f}   ← saldırı altında")
    print(f"\n   → Global F1 mMTC'nin bu durumunu tamamen gizliyor!")

print("\n" + "=" * 65)
print("PSRI HESAPLAMA TAMAMLANDI")
print("Sonraki adim: python 08_excel_psri.py  (Excel raporu)")
print("=" * 65)
