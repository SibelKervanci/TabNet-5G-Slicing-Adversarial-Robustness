"""
12_stronger_baselines.py
Güçlü Adversarial Baseline Karşılaştırması — Reviewer 1, Comment 4 için
------------------------------------------------------------------------
Saldırılar:
  1. AGE-Attention (Proposed)   — TabNet attention + surrogate transfer
  2. FGSM                       — Fast Gradient Sign Method
  3. PGD                        — Projected Gradient Descent (10 adım)
  4. C&W (L2)                   — Carlini & Wagner L2 saldırısı
  5. AutoAttack (APGD-CE)       — AutoAttack APGD-CE varyantı

Çalıştırma: python 12_stronger_baselines.py
Gereksinim: pip install autoattack
Çıktılar:   reports/18_stronger_baselines.png
            reports/stronger_baselines.csv
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

EPS  = 0.10
SEED = 42

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

# ── AutoAttack ────────────────────────────────────────────────────────────────
try:
    from autoattack import AutoAttack
    HAS_AA = True
    print("OK: autoattack mevcut")
except ImportError:
    HAS_AA = False
    print("UYARI: autoattack bulunamadi — pip install autoattack")
    print("       AutoAttack atlanacak, diger saldirilar calisacak")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VERİ
# ═══════════════════════════════════════════════════════════════════════════════
BASE = "CICIOT2023"

def find_label(df):
    for c in ["label","Label","attack_type","class","Class","target","Attack_type"]:
        if c in df.columns: return c
    return None

print("\n[1/5] Veri yukleniyor...")
df_tr = pd.read_csv(os.path.join(BASE,"train","train.csv"),       low_memory=False, nrows=200_000)
df_te = pd.read_csv(os.path.join(BASE,"test","test.csv"),         low_memory=False, nrows=50_000)
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
X_eval   = X_te[idx]
y_eval   = y_te[idx]
y_raw_eval = y_te_r[idx]

print(f"   Eval seti: {N_EVAL} ornek, {len(fc)} ozellik")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TABNET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] TabNet yukleniyor...")
tabnet = TabNetClassifier()
tabnet.load_model("models/tabnet_ciciot2023.zip")
y_clean  = tabnet.predict(X_eval)
acc_clean = accuracy_score(y_eval, y_clean)
print(f"   Clean acc: {acc_clean:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SURROGATE MLP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Surrogate MLP egitiliyor...")
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
# 4. ATTENTION MASK
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Attention + saldirilar...")
idx_va = rng.choice(len(X_va), min(2000,len(X_va)), replace=False)
expl,_ = tabnet.explain(X_va[idx_va])
feat_imp = np.mean(np.abs(expl), axis=0)
top5_att = np.argsort(feat_imp)[::-1][:5]
print(f"   TOP-5: {[fc[i] for i in top5_att]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SALDIRI FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════════════════

def to_tensor(X): return torch.from_numpy(X).to(DEVICE)

# --- AGE-Attention (tek adım, top-5) ---
def age_attack(X_np, top_idx, eps):
    X_t = to_tensor(X_np).requires_grad_(True)
    loss = -crit(surr(X_t), surr(X_t).argmax(1).detach())
    loss.backward()
    grad = X_t.grad.detach().cpu().numpy()
    mask = np.zeros_like(grad); mask[:,top_idx] = 1.0
    delta = eps * np.sign(grad) * mask
    return np.clip(X_np + delta, X_np-eps, X_np+eps).astype(np.float32)

# --- FGSM ---
def fgsm(X_np, eps):
    X_t = to_tensor(X_np).requires_grad_(True)
    loss = -crit(surr(X_t), surr(X_t).argmax(1).detach())
    loss.backward()
    return np.clip(X_np + eps*np.sign(X_t.grad.detach().cpu().numpy()),
                   X_np-eps, X_np+eps).astype(np.float32)

# --- PGD ---
def pgd(X_np, eps, steps=10):
    alpha = eps/steps
    X_adv = X_np.copy()
    for _ in range(steps):
        X_t = to_tensor(X_adv).requires_grad_(True)
        loss = -crit(surr(X_t), surr(X_t).argmax(1).detach())
        loss.backward()
        X_adv = X_adv + alpha*np.sign(X_t.grad.detach().cpu().numpy())
        X_adv = np.clip(X_adv, X_np-eps, X_np+eps).astype(np.float32)
    return X_adv

# --- C&W L2 ---
def cw_l2(X_np, eps, steps=200, lr=0.01, c=1.0):
    """
    Carlini & Wagner L2 saldırısı — surrogate MLP üzerinde.
    Hedef: yanlış sınıflandırma + küçük L2 pertürbasyon.
    """
    X_t   = to_tensor(X_np)
    y_t   = torch.from_numpy(
                surr(X_t).argmax(1).detach().cpu().numpy().astype(np.int64)
            ).to(DEVICE)

    # w: tanh alanında optimize edilecek delta
    w = torch.zeros_like(X_t, requires_grad=True, device=DEVICE)
    optimizer = optim.Adam([w], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        delta  = torch.tanh(w) * eps          # [-eps, eps] aralığında kalır
        X_adv  = X_t + delta
        logits = surr(X_adv)

        # Hedef sınıfın logit'ini düşür
        one_hot = torch.zeros_like(logits).scatter_(1, y_t.unsqueeze(1), 1.0)
        real    = (logits * one_hot).sum(1)
        other   = (logits * (1-one_hot) - one_hot*1e4).max(1).values
        f_loss  = torch.clamp(real - other + 0.0, min=0).mean()  # margin=0

        l2_loss = delta.norm(p=2, dim=1).mean()
        loss    = c * f_loss + l2_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        delta_final = torch.tanh(w) * eps
        X_adv_final = (X_t + delta_final).cpu().numpy()
    return np.clip(X_adv_final, X_np-eps, X_np+eps).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. BATCH ÇALIŞTIR
# ═══════════════════════════════════════════════════════════════════════════════
BATCH = 500

def batch_run(fn, X, **kwargs):
    parts = []
    for i in range(0, len(X), BATCH):
        parts.append(fn(X[i:i+BATCH], **kwargs))
    return np.vstack(parts)

print("   AGE-Attention...", end=" ", flush=True); t=time.time()
X_age  = batch_run(age_attack, X_eval, top_idx=top5_att, eps=EPS)
print(f"OK ({time.time()-t:.1f}s)")

print("   FGSM...", end=" ", flush=True); t=time.time()
X_fgsm = batch_run(fgsm, X_eval, eps=EPS)
print(f"OK ({time.time()-t:.1f}s)")

print("   PGD...", end=" ", flush=True); t=time.time()
X_pgd  = batch_run(pgd, X_eval, eps=EPS)
print(f"OK ({time.time()-t:.1f}s)")

print("   C&W L2 (200 adim)...", end=" ", flush=True); t=time.time()
X_cw   = batch_run(cw_l2, X_eval, eps=EPS)
print(f"OK ({time.time()-t:.1f}s)")

if HAS_AA:
    print("   AutoAttack (APGD-CE)...", end=" ", flush=True); t=time.time()
    # AutoAttack PyTorch modeli ister — surr kullan
    adversary = AutoAttack(surr, norm='Linf', eps=EPS, version='custom',
                           attacks_to_run=['apgd-ce'], device=DEVICE, verbose=False)
    X_eval_t = torch.from_numpy(X_eval).to(DEVICE)
    y_eval_t = torch.from_numpy(y_eval.astype(np.int64)).to(DEVICE)
    X_aa_t   = adversary.run_standard_evaluation(X_eval_t, y_eval_t, bs=BATCH)
    X_aa     = X_aa_t.cpu().numpy().astype(np.float32)
    print(f"OK ({time.time()-t:.1f}s)")
else:
    X_aa = None

# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEĞERLENDİRME
# ═══════════════════════════════════════════════════════════════════════════════
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

def evaluate(name, X_adv):
    y_adv   = tabnet.predict(X_adv)
    drop    = acc_clean - accuracy_score(y_eval, y_adv)
    l2      = float(np.mean(np.linalg.norm(X_adv-X_eval, axis=1)))
    fc_cnt  = float(np.mean((np.abs(X_adv-X_eval)>1e-6).sum(axis=1)))
    psri    = psri_slices(X_adv)
    urllc   = psri.get("URLLC", float("nan"))
    print(f"   {name:<28} drop={drop:.3f}  l2={l2:.3f}  feat={fc_cnt:.1f}  URLLC-PSRI={urllc:.3f}")
    return {"name":name,"drop":drop,"l2":l2,"feat":fc_cnt,
            **{f"psri_{k}":v for k,v in psri.items()}}

print("\nDegerlendirme (TabNet):")
rows = []
rows.append(evaluate("AGE-Attention (Proposed)", X_age))
rows.append(evaluate("FGSM",                     X_fgsm))
rows.append(evaluate("PGD (10 steps)",           X_pgd))
rows.append(evaluate("C&W L2",                   X_cw))
if X_aa is not None:
    rows.append(evaluate("AutoAttack (APGD-CE)", X_aa))

df = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. GRAFİK
# ═══════════════════════════════════════════════════════════════════════════════
short = {
    "AGE-Attention (Proposed)": "AGE-Att\n(Proposed)",
    "FGSM":                     "FGSM",
    "PGD (10 steps)":           "PGD\n(10 steps)",
    "C&W L2":                   "C&W L2",
    "AutoAttack (APGD-CE)":     "AutoAttack\n(APGD-CE)",
}
df["short"] = df["name"].map(short).fillna(df["name"])
COLORS = ["#E74C3C","#3498DB","#2ECC71","#9B59B6","#F39C12"]

fig, axes = plt.subplots(1,3, figsize=(18,6))
fig.suptitle(f"Stronger Baseline Comparison (ε={EPS}, n={N_EVAL})",
             fontsize=14, fontweight="bold")

for ax, col, ylabel, title in zip(
    axes,
    ["drop","l2","psri_URLLC"],
    ["Accuracy Drop (↑ more effective)",
     "L₂ Perturbation (↓ more stealthy)",
     "URLLC PSRI (↓ greater vulnerability)"],
    ["Attack Effectiveness","Stealthiness (L₂)","URLLC Slice Robustness"]
):
    vals = df[col].fillna(0).values if col in df.columns else np.zeros(len(df))
    bars = ax.bar(range(len(df)), vals,
                  color=COLORS[:len(df)], alpha=0.85, edgecolor="white", lw=1.2)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["short"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
out_fig = "reports/18_stronger_baselines.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nOK: {out_fig}")

out_csv = "reports/stronger_baselines.csv"
df.to_csv(out_csv, index=False)
print(f"OK: {out_csv}")

print("\n" + "="*70)
print("ÖZET")
print("="*70)
print(f"{'Variant':<28} {'Drop':>7} {'L2':>7} {'Feat':>7} {'URLLC-PSRI':>12}")
print("-"*70)
for _,r in df.iterrows():
    print(f"{r['name']:<28} {r['drop']:>7.4f} {r['l2']:>7.4f} "
          f"{r['feat']:>7.1f} {r.get('psri_URLLC',float('nan')):>12.4f}")
print("\nTAMAMLANDI")
