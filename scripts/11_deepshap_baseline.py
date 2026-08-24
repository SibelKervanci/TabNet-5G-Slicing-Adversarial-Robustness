"""
11_deepshap_baseline.py
DeepSHAP-guided adversarial attack baseline
AGE-Attention vs AGE-SHAP vs FGSM karşılaştırması

Çalıştırma: python 11_deepshap_baseline.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

try:
    import shap
    print("OK: SHAP hazir")
except ImportError:
    print("HATA: pip install shap"); exit(1)

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
    "eMBB":  ["DDoS-ICMP_Flood","DDoS-UDP_Flood","DDoS-TCP_Flood",
               "DDoS-PSHACK_Flood","DDoS-SYN_Flood","DDoS-RSTFINFlood",
               "DDoS-SynonymousIP_Flood","DDoS-ICMP_Fragmentation",
               "DDoS-UDP_Fragmentation","DDoS-ACK_Fragmentation",
               "DDoS-HTTP_Flood","DDoS-SlowLoris"],
    "URLLC": ["DoS-UDP_Flood","DoS-TCP_Flood","DoS-SYN_Flood","DoS-HTTP_Flood",
               "MITM-ArpSpoofing","DNS_Spoofing"],
    "mMTC":  ["Mirai-greeth_flood","Mirai-greip_flood","Mirai-udpplain",
               "Recon-HostDiscovery","Recon-OSScan","Recon-PortScan","Recon-PingSweep",
               "VulnerabilityScan","DictionaryBruteForce","BrowserHijacking",
               "SqlInjection","CommandInjection","XSS","Backdoor_Malware","Uploading_Attack"],
    "Benign":["BenignTraffic"],
}
CLASS_TO_SLICE = {cls:sl for sl,classes in SLICE_MAP.items() for cls in classes}

# ── VERİ ─────────────────────────────────────────────────────────────────────
BASE = "CICIOT2023"
print("\nVeri yukleniyor...")
df_train = pd.read_csv(os.path.join(BASE,"train","train.csv"), low_memory=False, nrows=200000)
df_test  = pd.read_csv(os.path.join(BASE,"test", "test.csv"), low_memory=False, nrows=50000)
df_val   = pd.read_csv(os.path.join(BASE,"validation","validation.csv"), low_memory=False, nrows=50000)

label_col    = "label"
feature_cols = [c for c in df_train.columns
                if c != label_col and df_train[c].dtype in [np.float64,np.int64,float,int]]

def prep(df):
    X = df[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0).values
    y = df[label_col].values
    return X, y

X_train_r, y_train_r = prep(df_train)
X_test_r,  y_test_r  = prep(df_test)
X_val_r,   y_val_r   = prep(df_val)

le = LabelEncoder().fit(np.concatenate([y_train_r, y_test_r, y_val_r]))
y_train = le.transform(y_train_r)
y_test  = le.transform(y_test_r)
y_val   = le.transform(y_val_r)

scaler  = StandardScaler().fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test  = scaler.transform(X_test_r)
X_val   = scaler.transform(X_val_r)

slice_labels = np.array([CLASS_TO_SLICE.get(c,"Unknown") for c in y_test_r])
print(f"OK: Train={len(X_train):,} | Test={len(X_test):,}")

# ── SURROGATE MLP ─────────────────────────────────────────────────────────────
print("\nSurrogate MLP egitiliyor...")
surrogate = MLPClassifier(
    hidden_layer_sizes=(256,128,64), activation="relu",
    max_iter=30, random_state=42,
    early_stopping=True, validation_fraction=0.1, verbose=False
)
surrogate.fit(X_train, y_train)
print(f"OK: Surrogate acc={accuracy_score(y_test, surrogate.predict(X_test)):.4f}")

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
            if i < len(sklearn_mlp.coefs_)-1:
                layers.append(nn.ReLU())
            in_size = out_size
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

torch_surr = SurrogateMLP(surrogate).to(DEVICE)
torch_surr.eval()

# ── TABNET YÜKLE ──────────────────────────────────────────────────────────────
print("\nTabNet yukleniyor...")
tabnet = TabNetClassifier()
tabnet.load_model("models/tabnet_ciciot2023.zip")

# ── SHAP — TOP-5 ÖZELLİK ─────────────────────────────────────────────────────
print("\nSHAP degerleri hesaplaniyor (DeepSHAP)...")
N_SHAP = 500
idx_bg  = np.random.choice(len(X_train), 200, replace=False)
idx_sh  = np.random.choice(len(X_test),  N_SHAP, replace=False)

background = torch.FloatTensor(X_train[idx_bg]).to(DEVICE)
test_data  = torch.FloatTensor(X_test[idx_sh]).to(DEVICE)

explainer   = shap.DeepExplainer(torch_surr, background)
shap_values = explainer.shap_values(test_data)

# Multi-class: shap_values liste — her sınıf için ayrı
# Mean absolute SHAP across all classes and samples
if isinstance(shap_values, list):
    # Liste formatı: her eleman (n_samples, n_features)
    shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    mean_shap = shap_abs.mean(axis=0)
elif shap_values.ndim == 3:
    # (n_samples, n_features, n_classes) formatı
    mean_shap = np.abs(shap_values).mean(axis=(0, 2))
elif shap_values.ndim == 2 and shap_values.shape == (len(feature_cols), len(le.classes_)):
    # (n_features, n_classes) formatı
    mean_shap = np.abs(shap_values).mean(axis=1)
else:
    mean_shap = np.abs(shap_values).mean(axis=0)

shap_series = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)

print("Top-10 SHAP ozellikleri:")
for i, (feat, val) in enumerate(shap_series.head(10).items(), 1):
    print(f"  {i:2}. {feat:<30} {val:.4f}")

top5_shap_idx = [feature_cols.index(f) for f in shap_series.head(5).index]

# ── TABNET ATTENTION TOP-5 (validation'dan) ───────────────────────────────────
print("\nTabNet attention mask yukleniyor...")
N_ATT   = 2000
idx_att = np.random.choice(len(X_val), N_ATT, replace=False)
explain_matrix, _ = tabnet.explain(X_val[idx_att])
feat_imp = pd.Series(explain_matrix.mean(axis=0), index=feature_cols).sort_values(ascending=False)
top5_att_idx = [feature_cols.index(f) for f in feat_imp.head(5).index]

print(f"Top-5 Attention: {feat_imp.head(5).index.tolist()}")
print(f"Top-5 SHAP:      {shap_series.head(5).index.tolist()}")

# Overlap
att_set  = set(feat_imp.head(5).index.tolist())
shap_set = set(shap_series.head(5).index.tolist())
overlap  = att_set & shap_set
print(f"Overlap (attention ∩ SHAP): {overlap} ({len(overlap)}/5)")

# ── SALDIRI FONKSİYONLARI ─────────────────────────────────────────────────────
def make_att_tensor(feat_idx, weights=None):
    mask = np.zeros(len(feature_cols))
    for i, idx in enumerate(feat_idx):
        mask[idx] = weights[i] if weights is not None else 1.0
    if mask.sum() > 0:
        mask = mask / mask.sum()
    return torch.FloatTensor(mask).to(DEVICE)

def transfer_attack(X, y, feat_idx, weights=None, epsilon=0.10, n_steps=10):
    att_t = make_att_tensor(feat_idx, weights)
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
            pert  = step * X_adv.grad.sign() * att_t.unsqueeze(0)
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

def l2_norm(X_o, X_a):
    return float(np.mean(np.linalg.norm(X_a - X_o, axis=1)))

def feat_changed(X_o, X_a, thr=1e-6):
    return float(np.mean((np.abs(X_a - X_o) > thr).sum(axis=1)))

# ── EVAL ─────────────────────────────────────────────────────────────────────
N_EVAL  = 8000
idx_e   = np.random.choice(len(X_test), N_EVAL, replace=False)
X_eval  = X_test[idx_e]
y_eval  = y_test[idx_e]
sl_eval = slice_labels[idx_e]
SLICES  = ["eMBB","URLLC","mMTC","Benign","GLOBAL"]

print("\nSaldirilar uretiliyor...")
att_weights  = feat_imp.head(5).values.tolist()
shap_weights = shap_series.head(5).values.tolist()

X_age_att  = transfer_attack(X_eval, y_eval, top5_att_idx,  att_weights,  epsilon=0.10)
X_age_shap = transfer_attack(X_eval, y_eval, top5_shap_idx, shap_weights, epsilon=0.10)
X_fgsm     = fgsm_transfer(X_eval, y_eval, epsilon=0.10)
print("OK: Saldirilar hazir")

# ── KARŞILAŞTIRMA ─────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("DeepSHAP vs AGE-Attention vs FGSM Karşılaştırması (ε=0.10)")
print("="*75)

y_clean = tabnet.predict(X_eval)
base_acc = accuracy_score(y_eval, y_clean)

results = {}
for name, X_adv in [("AGE-Attention (Proposed)", X_age_att),
                     ("AGE-SHAP (DeepSHAP)",      X_age_shap),
                     ("FGSM",                      X_fgsm)]:
    y_adv  = tabnet.predict(X_adv)
    drop   = base_acc - accuracy_score(y_eval, y_adv)
    l2     = l2_norm(X_eval, X_adv)
    fc     = feat_changed(X_eval, X_adv)

    psri_sl = {}
    for sl in SLICES:
        mask = np.ones(len(y_eval), dtype=bool) if sl=="GLOBAL" else (sl_eval==sl)
        if mask.sum()==0: continue
        ac = accuracy_score(y_eval[mask], y_clean[mask])
        aa = accuracy_score(y_eval[mask], y_adv[mask])
        psri_sl[sl] = aa/(ac+1e-9)

    results[name] = {"drop":drop,"l2":l2,"fc":fc,"psri":psri_sl}

print(f"\n{'Variant':<30} {'Drop':>8} {'L2':>8} {'Feat':>6} {'PSRI(URLLC)':>13} {'PSRI(mMTC)':>11}")
print("-"*80)
for name, res in results.items():
    u = res["psri"].get("URLLC", float("nan"))
    m = res["psri"].get("mMTC",  float("nan"))
    print(f"{name:<30} {res['drop']:>8.3f} {res['l2']:>8.3f} {res['fc']:>6.1f} {u:>13.3f} {m:>11.3f}")

# ── GRAFİK ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
names  = list(results.keys())
colors = ["#E74C3C","#8E44AD","#3498DB"]

ax = axes[0]
drops = [results[n]["drop"] for n in names]
bars  = ax.bar(range(len(names)), drops, color=colors, alpha=0.85)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(["AGE-Att\n(Proposed)","AGE-SHAP\n(DeepSHAP)","FGSM"], fontsize=9)
ax.set_ylabel("Accuracy Drop (↑ better)")
ax.set_title("Attack Effectiveness", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars, drops):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{v:.3f}", ha="center", fontsize=9)

ax = axes[1]
l2s  = [results[n]["l2"] for n in names]
bars = ax.bar(range(len(names)), l2s, color=colors, alpha=0.85)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(["AGE-Att\n(Proposed)","AGE-SHAP\n(DeepSHAP)","FGSM"], fontsize=9)
ax.set_ylabel("L₂ Perturbation (↓ stealthier)")
ax.set_title("Stealthiness", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars, l2s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{v:.3f}", ha="center", fontsize=9)

ax = axes[2]
sl_list = ["eMBB","URLLC","mMTC","Benign","GLOBAL"]
x = np.arange(len(sl_list))
w = 0.25
for i, (name, color) in enumerate(zip(names, colors)):
    vals = [results[name]["psri"].get(sl, 0) for sl in sl_list]
    ax.bar(x + (i-1)*w, vals, w, label=name.split("(")[0].strip(),
           color=color, alpha=0.85)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(sl_list, fontsize=9)
ax.set_ylabel("PSRI")
ax.set_title("Per-Slice PSRI", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig("reports/15_deepshap_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/15_deepshap_comparison.png")

# CSV
pd.DataFrame([
    {"Variant": name,
     "Acc_Drop": round(res["drop"],4),
     "L2":       round(res["l2"],4),
     "Feat":     round(res["fc"],1),
     "PSRI_eMBB":  round(res["psri"].get("eMBB",float("nan")),4),
     "PSRI_URLLC": round(res["psri"].get("URLLC",float("nan")),4),
     "PSRI_mMTC":  round(res["psri"].get("mMTC",float("nan")),4),
     "PSRI_GLOBAL":round(res["psri"].get("GLOBAL",float("nan")),4),
    }
    for name, res in results.items()
]).to_csv("reports/deepshap_comparison.csv", index=False)
print("OK: reports/deepshap_comparison.csv")

# Feature overlap raporu
print(f"\n--- Feature Overlap Report ---")
print(f"AGE-Attention top-5: {feat_imp.head(5).index.tolist()}")
print(f"AGE-SHAP top-5:      {shap_series.head(5).index.tolist()}")
print(f"Overlap: {overlap} ({len(overlap)}/5 features shared)")

print("\n" + "="*75)
print("TAMAMLANDI")
print("="*75)
