"""
07b_psri_transfer.py
PSRI hesaplama — Transfer tabanlı AGE ile güncellendi.
Çalıştırma: python 07b_psri_transfer.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OK: PyTorch {torch.__version__} | {DEVICE}")
except ImportError:
    print("HATA: pip install pytorch-tabnet"); exit(1)

# ── 5G DİLİM HARİTASI ────────────────────────────────────────────────────────
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
CLASS_TO_SLICE = {}
for sl, classes in SLICE_MAP.items():
    for cls in classes:
        CLASS_TO_SLICE[cls] = sl

# ── 1. MODEL YÜKLE ───────────────────────────────────────────────────────────
print("\nModel yukleniyor...")
model = TabNetClassifier()
model.load_model("models/tabnet_ciciot2023.zip")
model.network.eval()
print("OK: TabNet yuklendi")

# ── 2. VERİ YÜKLE ────────────────────────────────────────────────────────────
BASE = "CICIOT2023"
print("Veri yukleniyor...")
df_train = pd.read_csv(os.path.join(BASE, "train", "train.csv"),
                       low_memory=False, nrows=80000)
df_test  = pd.read_csv(os.path.join(BASE, "test",  "test.csv"),
                       low_memory=False, nrows=50000)

label_col    = "label"
feature_cols = [c for c in df_test.columns
                if c != label_col and df_test[c].dtype in [np.float64, np.int64, float, int]]

def prep(df):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df[label_col].values
    return X, y

X_train_r, y_train_r = prep(df_train)
X_test_r,  y_test_r  = prep(df_test)

le = LabelEncoder().fit(np.concatenate([y_train_r, y_test_r]))
y_train = le.transform(y_train_r)
y_test  = le.transform(y_test_r)
class_names = le.classes_

scaler  = StandardScaler().fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test  = scaler.transform(X_test_r)

slice_labels = np.array([CLASS_TO_SLICE.get(c, "Unknown") for c in y_test_r])
print(f"OK: Test={len(X_test):,} | Dilim dagilimi:")
for sl in ["eMBB","URLLC","mMTC","Benign"]:
    cnt = (slice_labels == sl).sum()
    print(f"   {sl:<8} {cnt:>6,}  ({cnt/len(X_test)*100:.1f}%)")

# ── 3. SURROGATE MLP ─────────────────────────────────────────────────────────
print("\nSurrogate MLP egitiliyor...")
surrogate = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    max_iter=50,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
surrogate.fit(X_train, y_train)
print(f"OK: Surrogate accuracy: {accuracy_score(y_test, surrogate.predict(X_test)):.4f}")

# PyTorch surrogate
class SurrogateMLP(nn.Module):
    def __init__(self, sklearn_mlp):
        super().__init__()
        layers = []
        in_size = len(feature_cols)
        for i, (coef, intercept) in enumerate(zip(sklearn_mlp.coefs_, sklearn_mlp.intercepts_)):
            out_size = coef.shape[1]
            layer = nn.Linear(in_size, out_size)
            layer.weight = nn.Parameter(torch.FloatTensor(coef.T))
            layer.bias   = nn.Parameter(torch.FloatTensor(intercept))
            layers.append(layer)
            if i < len(sklearn_mlp.coefs_) - 1:
                layers.append(nn.ReLU())
            in_size = out_size
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

torch_surr = SurrogateMLP(surrogate).to(DEVICE)
torch_surr.eval()
print("OK: PyTorch surrogate hazir")

# ── 4. ATTENTION ÖZELLIKLERI ─────────────────────────────────────────────────
top_feats = []
if os.path.exists("reports/top_attention_features.csv"):
    top_feats    = pd.read_csv("reports/top_attention_features.csv", header=None)[0].tolist()
    top_feat_idx = [feature_cols.index(f) for f in top_feats if f in feature_cols]
else:
    top_feat_idx = list(range(5))

# Attention ağırlık maskesi
att_mask = np.zeros(len(feature_cols))
if os.path.exists("reports/full_feature_importance.csv"):
    fi = pd.read_csv("reports/full_feature_importance.csv", index_col=0).squeeze()
    for idx in top_feat_idx:
        fname = feature_cols[idx]
        if fname in fi.index:
            att_mask[idx] = fi[fname]
else:
    for idx in top_feat_idx:
        att_mask[idx] = 1.0
if att_mask.sum() > 0:
    att_mask = att_mask / att_mask.sum()
att_tensor = torch.FloatTensor(att_mask).to(DEVICE)

print(f"OK: AGE hedefleri: {top_feats}")

# ── 5. SALDIRI FONKSİYONLARI ─────────────────────────────────────────────────
def transfer_age(X, y, epsilon=0.10, n_steps=10):
    step  = epsilon / n_steps
    X_t   = torch.FloatTensor(X).to(DEVICE)
    y_t   = torch.LongTensor(y).to(DEVICE)
    X_adv = X_t.clone().detach()
    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        out  = torch_surr(X_adv)
        loss = F.cross_entropy(out, y_t)
        loss.backward()
        with torch.no_grad():
            pert  = step * X_adv.grad.sign() * att_tensor.unsqueeze(0)
            X_adv = X_adv + pert
            delta = torch.clamp(X_adv - X_t, -epsilon, epsilon)
            X_adv = (X_t + delta).detach()
    return X_adv.cpu().numpy()

def fgsm_transfer(X, y, epsilon=0.10):
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y).to(DEVICE)
    X_t.requires_grad_(True)
    out  = torch_surr(X_t)
    loss = F.cross_entropy(out, y_t)
    loss.backward()
    return (X_t + epsilon * X_t.grad.sign()).detach().cpu().numpy()

def l2_norm(X_orig, X_adv):
    return float(np.mean(np.linalg.norm(X_adv - X_orig, axis=1)))

def feat_changed(X_orig, X_adv, thr=1e-6):
    return float(np.mean((np.abs(X_adv - X_orig) > thr).sum(axis=1)))

# ── 6. PSRI HESAPLAMA ────────────────────────────────────────────────────────
print("\nPSRI hesaplaniyor (Transfer AGE ile)...")

N_EVAL = min(8000, len(X_test))
idx    = np.random.choice(len(X_test), N_EVAL, replace=False)
X_eval      = X_test[idx]
y_eval      = y_test[idx]
y_str_eval  = y_test_r[idx]
sl_eval     = slice_labels[idx]

# Tahminler
y_clean = model.predict(X_eval)
X_age   = transfer_age(X_eval, y_eval, epsilon=0.10)
X_fgsm  = fgsm_transfer(X_eval, y_eval, epsilon=0.10)
y_age   = model.predict(X_age)
y_fgsm  = model.predict(X_fgsm)

# L2 ve feature sayısı
l2_age  = l2_norm(X_eval, X_age)
l2_fgsm = l2_norm(X_eval, X_fgsm)
fc_age  = feat_changed(X_eval, X_age)
fc_fgsm = feat_changed(X_eval, X_fgsm)

print(f"\nGizlilik karsilastirmasi (ε=0.10):")
print(f"  AGE  L2={l2_age:.3f}  Feat={fc_age:.1f}/46")
print(f"  FGSM L2={l2_fgsm:.3f}  Feat={fc_fgsm:.1f}/46")

results = {}
slices  = ["eMBB","URLLC","mMTC","Benign","GLOBAL"]

for sl in slices:
    mask = np.ones(len(y_eval), dtype=bool) if sl == "GLOBAL" else (sl_eval == sl)
    if mask.sum() == 0:
        continue
    yc = y_clean[mask]; ya = y_age[mask]; yf = y_fgsm[mask]; yt = y_eval[mask]

    acc_clean = accuracy_score(yt, yc)
    acc_age   = accuracy_score(yt, ya)
    acc_fgsm  = accuracy_score(yt, yf)
    f1_clean  = f1_score(yt, yc, average="weighted", zero_division=0)
    f1_age    = f1_score(yt, ya, average="weighted", zero_division=0)
    f1_fgsm   = f1_score(yt, yf, average="weighted", zero_division=0)
    psri_age  = acc_age  / (acc_clean + 1e-9)
    psri_fgsm = acc_fgsm / (acc_clean + 1e-9)

    results[sl] = {
        "n": mask.sum(),
        "acc_clean": acc_clean, "acc_age": acc_age,   "acc_fgsm": acc_fgsm,
        "f1_clean":  f1_clean,  "f1_age":  f1_age,    "f1_fgsm":  f1_fgsm,
        "psri_age":  psri_age,  "psri_fgsm": psri_fgsm,
        "drop_age":  acc_clean - acc_age,
        "drop_fgsm": acc_clean - acc_fgsm,
    }

# ── 7. RAPOR ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PER-SLICE ROBUSTNESS INDEX (PSRI) — Transfer AGE Sonuclari")
print("="*70)
print(f"\n{'Dilim':<10} {'N':>6} {'Temiz':>8} {'AGE':>8} {'FGSM':>8} "
      f"{'PSRI(AGE)':>10} {'PSRI(FGSM)':>11} {'Fark':>8}")
print("-"*75)

for sl in slices:
    if sl not in results:
        continue
    r    = results[sl]
    diff = r["psri_fgsm"] - r["psri_age"]
    flag = " <- AGE daha etkili" if r["psri_age"] < r["psri_fgsm"] - 0.05 else ""
    print(f"{sl:<10} {r['n']:>6,} {r['acc_clean']:>8.3f} {r['acc_age']:>8.3f} "
          f"{r['acc_fgsm']:>8.3f} {r['psri_age']:>10.3f} {r['psri_fgsm']:>11.3f} "
          f"{diff:>+8.3f}{flag}")

print("\nSW-F1:")
print(f"{'Dilim':<10} {'F1 Temiz':>10} {'F1 AGE':>10} {'F1 FGSM':>10}")
print("-"*45)
for sl in slices:
    if sl not in results:
        continue
    r = results[sl]
    print(f"{sl:<10} {r['f1_clean']:>10.3f} {r['f1_age']:>10.3f} {r['f1_fgsm']:>10.3f}")

# ── 8. GÖRSELLEŞTİRME ───────────────────────────────────────────────────────
slice_order = [s for s in slices if s in results]
psri_age_v  = [results[s]["psri_age"]  for s in slice_order]
psri_fgsm_v = [results[s]["psri_fgsm"] for s in slice_order]
acc_clean_v = [results[s]["acc_clean"] for s in slice_order]
acc_age_v   = [results[s]["acc_age"]   for s in slice_order]
acc_fgsm_v  = [results[s]["acc_fgsm"]  for s in slice_order]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("")
w = 0.35
x = np.arange(len(slice_order))

# Sol: PSRI
ax = axes[0]
bars1 = ax.bar(x - w/2, psri_age_v,  w, label="AGE (Transfer)", color="#E74C3C", alpha=0.85)
bars2 = ax.bar(x + w/2, psri_fgsm_v, w, label="FGSM",           color="#3498DB", alpha=0.85)
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

# Orta: Accuracy
ax = axes[1]
ax.plot(slice_order, acc_clean_v, "g-o",  lw=2.5, ms=8, label="Clean")
ax.plot(slice_order, acc_age_v,   "r-s",  lw=2.5, ms=8, label="AGE (Transfer)")
ax.plot(slice_order, acc_fgsm_v,  "b--^", lw=2,   ms=7, label="FGSM")
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
ax.set_title("")
ax.legend(); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Sağ: F1 düşüşü
ax = axes[2]
f1_drop_age  = [results[s]["f1_clean"] - results[s]["f1_age"]  for s in slice_order]
f1_drop_fgsm = [results[s]["f1_clean"] - results[s]["f1_fgsm"] for s in slice_order]
ax.bar(x - w/2, f1_drop_age,  w, label="AGE (Transfer)", color="#E74C3C", alpha=0.85)
ax.bar(x + w/2, f1_drop_fgsm, w, label="FGSM",           color="#3498DB", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(slice_order, fontsize=10)
ax.set_ylabel("F1 Drop (higher = worse)")
ax.set_title("")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("reports/11_psri_transfer.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/11_psri_transfer.png")

# CSV
pd.DataFrame([
    {"Dilim": sl,
     "N": results[sl]["n"],
     "Acc_Clean":  round(results[sl]["acc_clean"],  4),
     "Acc_AGE":    round(results[sl]["acc_age"],    4),
     "Acc_FGSM":   round(results[sl]["acc_fgsm"],   4),
     "F1_Clean":   round(results[sl]["f1_clean"],   4),
     "F1_AGE":     round(results[sl]["f1_age"],     4),
     "F1_FGSM":    round(results[sl]["f1_fgsm"],    4),
     "PSRI_AGE":   round(results[sl]["psri_age"],   4),
     "PSRI_FGSM":  round(results[sl]["psri_fgsm"],  4),
     "Drop_AGE":   round(results[sl]["drop_age"],   4),
     "Drop_FGSM":  round(results[sl]["drop_fgsm"],  4),
     "L2_AGE":     round(l2_age, 4),
     "L2_FGSM":    round(l2_fgsm, 4),
     "Feat_AGE":   round(fc_age, 1),
     "Feat_FGSM":  round(fc_fgsm, 1),
    }
    for sl in slice_order
]).to_csv("reports/psri_transfer_sonuclari.csv", index=False)
print("OK: reports/psri_transfer_sonuclari.csv")

print("\n" + "="*70)
print("="*70)
