"""
13_topk_sensitivity.py
Top-K Özellik Sayısı Duyarlılık Analizi — Reviewer 1, Comment 6 için
----------------------------------------------------------------------
K = 1, 3, 5, 7, 10, 15, 20 için AGE-Attention saldırısını çalıştırır.
Her K için: Acc Drop, L2, Feat sayısı, URLLC PSRI ölçülür.

Çalıştırma: python 13_topk_sensitivity.py
Çıktılar:   reports/19_topk_sensitivity.png
            reports/topk_sensitivity.csv
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)

EPS   = 0.10
SEED  = 42
K_VALUES = [1, 3, 5, 7, 10, 15, 20]

SLICES = {
    "eMBB":  ["DDoS-RSTFINFlood","DDoS-PSHACK_Flood","DDoS-SYN_Flood",
               "DDoS-UDP_Flood","DDoS-TCP_Flood","DDoS-ICMP_Flood",
               "DDoS-SlowLoris","DDoS-HTTP_Flood","DoS-UDP_Flood",
               "DoS-SYN_Flood","DoS-TCP_Flood","DoS-HTTP_Flood","BenignTraffic"],
    "URLLC": ["MITM-ArpSpoofing","DNS_Spoofing",
               "DoS-UDP_Flood","DoS-SYN_Flood","DoS-TCP_Flood","DoS-HTTP_Flood"],
    "mMTC":  ["Mirai-greeth_flood","Mirai-greip_flood","Mirai-udpplain",
               "Recon-HostDiscovery","Recon-OSScan","Recon-PortScan",
               "VulnerabilityScan","SqlInjection","CommandInjection",
               "XSS","Backdoor_Malware","Uploading_Attack","BrowserHijacking",
               "DictionaryBruteForce"],
}

# ── PyTorch / TabNet ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from pytorch_tabnet.tab_model import TabNetClassifier
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OK: PyTorch {torch.__version__} | device={DEVICE}")
except ImportError as e:
    print(f"HATA: {e}"); raise SystemExit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VERİ
# ═══════════════════════════════════════════════════════════════════════════════
BASE = "CICIOT2023"

def find_label(df):
    for c in ["label","Label","attack_type","class","Class","target","Attack_type"]:
        if c in df.columns: return c
    return None

print("\n[1/4] Veri yukleniyor...")
df_tr = pd.read_csv(os.path.join(BASE,"train","train.csv"),           low_memory=False, nrows=200_000)
df_te = pd.read_csv(os.path.join(BASE,"test","test.csv"),             low_memory=False, nrows=50_000)
df_va = pd.read_csv(os.path.join(BASE,"validation","validation.csv"), low_memory=False, nrows=50_000)

lc = find_label(df_tr)
fc = [c for c in df_tr.columns if c != lc and df_tr[c].dtype in [np.float64,np.int64,float,int]]

def prep(df):
    X = df[fc].replace([np.inf,-np.inf],np.nan).fillna(0).values
    return X, df[lc].values

X_tr_r,y_tr_r = prep(df_tr)
X_te_r,y_te_r = prep(df_te)
X_va_r,y_va_r = prep(df_va)

le = LabelEncoder().fit(np.concatenate([y_tr_r,y_te_r,y_va_r]))
y_tr = le.transform(y_tr_r)
y_te = le.transform(y_te_r)

sc = StandardScaler().fit(X_tr_r)
X_tr = sc.transform(X_tr_r).astype(np.float32)
X_te = sc.transform(X_te_r).astype(np.float32)
X_va = sc.transform(X_va_r).astype(np.float32)

N_EVAL = min(2000, len(X_te))
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_te), N_EVAL, replace=False)
X_eval     = X_te[idx]
y_eval     = y_te[idx]
y_raw_eval = y_te_r[idx]
print(f"   Eval seti: {N_EVAL} ornek, {len(fc)} ozellik")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TABNET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] TabNet yukleniyor...")
tabnet = TabNetClassifier()
tabnet.load_model("models/tabnet_ciciot2023.zip")
y_clean   = tabnet.predict(X_eval)
acc_clean = accuracy_score(y_eval, y_clean)
print(f"   Clean acc: {acc_clean:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SURROGATE MLP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Surrogate MLP egitiliyor...")
N_FEAT    = X_tr.shape[1]
N_CLASSES = len(le.classes_)

class SurrogateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEAT,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64),    nn.ReLU(),
            nn.Linear(64,N_CLASSES)
        )
    def forward(self,x): return self.net(x)

surr = SurrogateMLP().to(DEVICE)
opt  = optim.Adam(surr.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

Xt = torch.from_numpy(X_tr).to(DEVICE)
yt = torch.from_numpy(y_tr.astype(np.int64)).to(DEVICE)
loader = DataLoader(TensorDataset(Xt,yt), batch_size=512, shuffle=True)

surr.train()
for ep in range(30):
    for xb,yb in loader:
        opt.zero_grad(); crit(surr(xb),yb).backward(); opt.step()
    if (ep+1) % 10 == 0:
        surr.eval()
        with torch.no_grad():
            va = (surr(torch.from_numpy(X_eval).to(DEVICE)).argmax(1).cpu().numpy()==y_eval).mean()
        surr.train()
        print(f"   Epoch {ep+1:2d}/30  val_acc={va:.4f}")
surr.eval()
print("   OK: Surrogate hazir")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ATTENTION MASK — TÜM SIRALI ÖZELLİKLER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Attention mask hesaplaniyor...")
idx_va = rng.choice(len(X_va), min(2000,len(X_va)), replace=False)
expl,_ = tabnet.explain(X_va[idx_va])
feat_imp = np.mean(np.abs(expl), axis=0)
sorted_idx = np.argsort(feat_imp)[::-1]   # azalan sırada tüm özellikler
print(f"   TOP-5: {[fc[i] for i in sorted_idx[:5]]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SALDIRI & DEĞERLENDİRME
# ═══════════════════════════════════════════════════════════════════════════════
def to_tensor(X): return torch.from_numpy(X).to(DEVICE)

def age_attack(X_np, top_idx, eps):
    """AGE-Attention: tek adım, sadece top_idx özellikleri perturbe eder."""
    X_t = to_tensor(X_np).requires_grad_(True)
    loss = -crit(surr(X_t), surr(X_t).argmax(1).detach())
    loss.backward()
    grad = X_t.grad.detach().cpu().numpy()
    mask = np.zeros_like(grad)
    mask[:, top_idx] = 1.0
    delta = eps * np.sign(grad) * mask
    return np.clip(X_np + delta, X_np - eps, X_np + eps).astype(np.float32)

def psri_slices(X_adv):
    res = {}
    for sl, classes in SLICES.items():
        ids  = [i for i,c in enumerate(le.classes_) if c in classes]
        mask = np.isin(y_eval, ids)
        if mask.sum() < 5:
            res[sl] = float("nan"); continue
        ac = accuracy_score(y_eval[mask], tabnet.predict(X_eval[mask]))
        aa = accuracy_score(y_eval[mask], tabnet.predict(X_adv[mask]))
        res[sl] = aa/ac if ac > 1e-9 else float("nan")
    return res

BATCH = 500
def batch_run(fn, X, **kwargs):
    parts = []
    for i in range(0, len(X), BATCH):
        parts.append(fn(X[i:i+BATCH], **kwargs))
    return np.vstack(parts)

rows = []
print(f"\n{'K':>4} {'Drop':>7} {'L2':>7} {'Feat':>6} {'URLLC-PSRI':>12}  Features")
print("-"*75)

for K in K_VALUES:
    top_k = sorted_idx[:K]
    t = time.time()
    X_adv = batch_run(age_attack, X_eval, top_idx=top_k, eps=EPS)

    y_adv   = tabnet.predict(X_adv)
    drop    = acc_clean - accuracy_score(y_eval, y_adv)
    l2      = float(np.mean(np.linalg.norm(X_adv - X_eval, axis=1)))
    fc_cnt  = float(np.mean((np.abs(X_adv - X_eval) > 1e-6).sum(axis=1)))
    psri    = psri_slices(X_adv)
    urllc   = psri.get("URLLC", float("nan"))
    feat_names = ", ".join([fc[i] for i in top_k[:5]])
    if K > 5: feat_names += f" (+{K-5} more)"

    print(f"{K:>4} {drop:>7.4f} {l2:>7.4f} {fc_cnt:>6.1f} {urllc:>12.4f}  {feat_names}  ({time.time()-t:.1f}s)")

    rows.append({
        "K": K,
        "drop": drop,
        "l2": l2,
        "feat": fc_cnt,
        "psri_eMBB":  psri.get("eMBB",  float("nan")),
        "psri_URLLC": urllc,
        "psri_mMTC":  psri.get("mMTC",  float("nan")),
        "top_features": feat_names,
    })

df = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRAFİK
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Top-K Feature Sensitivity Analysis (ε={EPS}, n={N_EVAL})",
             fontsize=14, fontweight="bold")

metrics = [
    ("drop",       "Accuracy Drop (↑ more effective)",      "#E74C3C"),
    ("l2",         "L₂ Perturbation (↓ more stealthy)",     "#3498DB"),
    ("feat",       "Features Modified (↓ more stealthy)",   "#2ECC71"),
    ("psri_URLLC", "URLLC PSRI (↓ greater vulnerability)",  "#9B59B6"),
]

for ax, (col, ylabel, color) in zip(axes.flatten(), metrics):
    vals = df[col].values
    ax.plot(df["K"], vals, "o-", color=color, lw=2.5, ms=8)
    ax.fill_between(df["K"], vals, alpha=0.15, color=color)

    # K=5 noktasını vurgula
    k5_row = df[df["K"]==5]
    if not k5_row.empty:
        ax.axvline(x=5, color="gray", linestyle="--", lw=1.2, alpha=0.7)
        ax.scatter([5], [k5_row[col].values[0]], color="red", zorder=5,
                   s=120, label="K=5 (proposed)")
        ax.legend(fontsize=9)

    for k, v in zip(df["K"], vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Number of Top-K Features Perturbed", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(df["K"])
    ax.grid(True, alpha=0.3)
    ax.set_title(ylabel.split("(")[0].strip(), fontweight="bold")

plt.tight_layout()
out_fig = "reports/19_topk_sensitivity.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nOK: {out_fig}")

out_csv = "reports/topk_sensitivity.csv"
df.to_csv(out_csv, index=False)
print(f"OK: {out_csv}")

print("\n" + "="*75)
print("ÖZET — Top-K Duyarlılık Analizi")
print("="*75)
print(f"{'K':>4} {'Drop':>8} {'L2':>8} {'Feat':>8} {'URLLC-PSRI':>12}")
print("-"*50)
for _, r in df.iterrows():
    marker = " ◄ proposed" if r["K"] == 5 else ""
    print(f"{int(r['K']):>4} {r['drop']:>8.4f} {r['l2']:>8.4f} "
          f"{r['feat']:>8.1f} {r['psri_URLLC']:>12.4f}{marker}")
print("\nTAMAMLANDI")
