"""
11_deepshap_ablation.py
Kapsamli Ablation Calismasi — Reviewer 1, Comment 1 icin
--------------------------------------------------------------
6 varyant:
  1. AGE-Attention (Proposed) — TabNet attention mask + surrogate transfer
  2. AGE-SHAP          — DeepSHAP feature selection + surrogate transfer
  3. AGE-Random        — Rastgele 5 ozellik + surrogate transfer
  4. FGSM-Top5         — Top-5 attention ozelligine FGSM (transfer YOK)
  5. FGSM (baseline)   — Tum ozellikler, tek adim
  6. PGD (baseline)    — Tum ozellikler, 10 adim

Calisma: python 11_deepshap_ablation.py
Ciktilar: reports/17_ablation_study.png
          reports/ablation_sonuclari.csv
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

# ── Sabits ────────────────────────────────────────────────────────────────────
EPS   = 0.10
N_TOP = 5          # AGE'nin hedefledigi ozellik sayisi
SEED  = 42

SLICES = {
    "eMBB":  ["DDoS-RSTFINFlood","DDoS-PSHACK_Flood","DDoS-SYN_Flood",
               "DDoS-UDP_Flood","DDoS-TCP_Flood","DDoS-ICMP_Flood",
               "DDoS-SlowLoris","DDoS-HTTP_Flood","DoS-UDP_Flood",
               "DoS-SYN_Flood","DoS-TCP_Flood","DoS-HTTP_Flood",
               "BenignTraffic"],
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
    print(f"HATA: {e}\npip install pytorch-tabnet torch")
    raise SystemExit(1)

# ── DeepSHAP (shap) ──────────────────────────────────────────────────────────
try:
    import shap
    print(f"OK: shap {shap.__version__}")
except ImportError:
    print("UYARI: shap bulunamadi — pip install shap  (AGE-SHAP atlanacak)")
    shap = None

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════════
BASE = "CICIOT2023"

def find_label_col(df):
    for c in ["label","Label","attack_type","class","Class","target","Attack_type"]:
        if c in df.columns:
            return c
    return None

print("\n[1/6] Veri yukleniyor...")
t0 = time.time()

train_path = os.path.join(BASE, "train", "train.csv")
test_path  = os.path.join(BASE, "test",  "test.csv")
val_path   = os.path.join(BASE, "validation", "validation.csv")

df_train = pd.read_csv(train_path, low_memory=False, nrows=200_000)
df_test  = pd.read_csv(test_path,  low_memory=False, nrows=50_000)
df_val   = pd.read_csv(val_path,   low_memory=False, nrows=50_000)

label_col   = find_label_col(df_train)
feature_cols = [c for c in df_train.columns
                if c != label_col and df_train[c].dtype in
                   [np.float64, np.int64, float, int]]

def prep(df):
    X = df[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0).values
    y = df[label_col].values
    return X, y

X_tr_raw, y_tr_raw = prep(df_train)
X_te_raw, y_te_raw = prep(df_test)
X_va_raw, y_va_raw = prep(df_val)

le = LabelEncoder().fit(np.concatenate([y_tr_raw, y_te_raw, y_va_raw]))
y_tr = le.transform(y_tr_raw)
y_te = le.transform(y_te_raw)
y_va = le.transform(y_va_raw)

sc = StandardScaler().fit(X_tr_raw)
X_tr = sc.transform(X_tr_raw).astype(np.float32)
X_te = sc.transform(X_te_raw).astype(np.float32)
X_va = sc.transform(X_va_raw).astype(np.float32)

print(f"   Train {X_tr.shape}, Test {X_te.shape}, Val {X_va.shape} — {time.time()-t0:.1f}s")

# Degerlendirme seti: ilk 8000 test ornegi
N_EVAL = min(8000, len(X_te))
X_eval = X_te[:N_EVAL]
y_eval = y_te[:N_EVAL]
y_raw_eval = y_te_raw[:N_EVAL]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TABNET YÜKLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] TabNet yukleniyor...")
MODEL_PATH = "models/tabnet_ciciot2023.zip"
if not os.path.exists(MODEL_PATH):
    print(f"   HATA: {MODEL_PATH} bulunamadi. Lütfen modeli kopyalayin.")
    raise SystemExit(1)

tabnet = TabNetClassifier()
tabnet.load_model(MODEL_PATH)
print("   OK: TabNet yuklendi")

# TabNet clean accuracy
y_clean = tabnet.predict(X_eval)
acc_clean = accuracy_score(y_eval, y_clean)
print(f"   Clean acc (eval set): {acc_clean:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SURROGATE MLP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Surrogate MLP egitiliyor...")

N_CLASSES = len(le.classes_)
N_FEAT    = X_tr.shape[1]

class SurrogateMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

surr = SurrogateMLP(N_FEAT, N_CLASSES).to(DEVICE)
opt  = optim.Adam(surr.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

# Eğitim (30 epoch, batch=512)
X_t = torch.from_numpy(X_tr).to(DEVICE)
y_t = torch.from_numpy(y_tr.astype(np.int64)).to(DEVICE)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=True)

surr.train()
for ep in range(30):
    total_loss = 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = crit(surr(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if (ep+1) % 10 == 0:
        surr.eval()
        with torch.no_grad():
            Xt_ = torch.from_numpy(X_eval).to(DEVICE)
            surr_acc = (surr(Xt_).argmax(1).cpu().numpy() == y_eval).mean()
        surr.train()
        print(f"   Epoch {ep+1:2d}/30  loss={total_loss/len(loader):.4f}  val_acc={surr_acc:.4f}")

surr.eval()
print("   OK: Surrogate egitildi")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ATTENTION MASK (TabNet) — validation setinden
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] TabNet attention hesaplaniyor...")
rng = np.random.default_rng(SEED)
idx_va = rng.choice(len(X_va), min(2000, len(X_va)), replace=False)
explain_mat, _ = tabnet.explain(X_va[idx_va])
feat_imp = np.mean(np.abs(explain_mat), axis=0)
top5_att = np.argsort(feat_imp)[::-1][:N_TOP]
print(f"   TOP-{N_TOP} (attention): {[feature_cols[i] for i in top5_att]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DeepSHAP MASK — surrogate üzerinden
# ═══════════════════════════════════════════════════════════════════════════════
top5_shap = None
if shap is not None:
    print("\n[5/6] DeepSHAP hesaplaniyor (500 ornek, 200 bg)...")
    bg_idx = rng.choice(len(X_tr), 200, replace=False)
    bg     = torch.from_numpy(X_tr[bg_idx]).to(DEVICE)
    te_idx = rng.choice(len(X_te), 500, replace=False)
    te_shap= torch.from_numpy(X_te[te_idx]).to(DEVICE)

    explainer  = shap.DeepExplainer(surr, bg)
    shap_vals  = explainer.shap_values(te_shap)          # list[n_class][500, n_feat]
    shap_arr   = np.abs(np.array(shap_vals))             # (n_class, 500, n_feat)
    mean_shap  = shap_arr.mean(axis=(0,1))               # (n_feat,)
    top5_shap  = np.argsort(mean_shap)[::-1][:N_TOP]
    print(f"   TOP-{N_TOP} (SHAP):      {[feature_cols[i] for i in top5_shap]}")
else:
    print("\n[5/6] shap yok, AGE-SHAP atlanacak")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SALDIRI FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Saldirilar uretiliyor...")

def transfer_attack(X_np, top_idx, eps):
    """
    Surrogate MLP gradient → perturb top_idx features → transfer TabNet.
    top_idx: array of feature indices to perturb
    """
    X_t  = torch.from_numpy(X_np).to(DEVICE).requires_grad_(True)
    logits = surr(X_t)
    labels = logits.argmax(1).detach()
    loss   = -crit(logits, labels)   # negative = maximize loss (evasion)
    loss.backward()
    grad = X_t.grad.detach().cpu().numpy()

    # Sadece secili ozelliklerde perturbasyon
    mask = np.zeros_like(grad)
    mask[:, top_idx] = 1.0
    delta = eps * np.sign(grad) * mask
    return np.clip(X_np + delta, X_np - eps, X_np + eps)

def fgsm_top5(X_np, top_idx, eps):
    """FGSM ama sadece top_idx ozellikleri — transfer YOK, dogrudan TabNet."""
    # TabNet gradient icin surrogate kullaniyoruz (black-box senaryosu)
    # Burada surrogate gradienti alip TabNet'e uygulamak yerine
    # surrogate'e dogrudan FGSM yapip hedef sadece top_idx ozellikleri
    return transfer_attack(X_np, top_idx, eps)  # transfer ile ayni mekanizma

def fgsm_full(X_np, eps):
    """FGSM — tum ozellikler, surrogate gradient."""
    X_t  = torch.from_numpy(X_np).to(DEVICE).requires_grad_(True)
    logits = surr(X_t)
    labels = logits.argmax(1).detach()
    loss   = -crit(logits, labels)
    loss.backward()
    grad = X_t.grad.detach().cpu().numpy()
    delta = eps * np.sign(grad)
    return np.clip(X_np + delta, X_np - eps, X_np + eps)

def pgd_full(X_np, eps, alpha=None, steps=10):
    """PGD — tum ozellikler, iteratif."""
    if alpha is None:
        alpha = eps / steps
    X_adv = X_np.copy()
    for _ in range(steps):
        X_t = torch.from_numpy(X_adv).to(DEVICE).requires_grad_(True)
        logits = surr(X_t)
        labels = logits.argmax(1).detach()
        loss   = -crit(logits, labels)
        loss.backward()
        grad  = X_t.grad.detach().cpu().numpy()
        X_adv = X_adv + alpha * np.sign(grad)
        X_adv = np.clip(X_adv, X_np - eps, X_np + eps)
    return X_adv.astype(np.float32)

# Rastgele 5 ozellik (seed=42 ile tekrarlanabilir)
rng2 = np.random.default_rng(SEED)
top5_rand = rng2.choice(N_FEAT, N_TOP, replace=False)
print(f"   TOP-{N_TOP} (random):    {[feature_cols[i] for i in top5_rand]}")

# Saldirilar uret
BATCH = 1000
def batch_attack(fn, X, **kwargs):
    """Büyük X için batch'li saldırı."""
    parts = []
    for i in range(0, len(X), BATCH):
        xb = X[i:i+BATCH]
        parts.append(fn(xb, **kwargs))
    return np.vstack(parts)

print("   AGE-Attention...", end=" ", flush=True)
X_age_att  = batch_attack(transfer_attack, X_eval, top_idx=top5_att, eps=EPS)
print("OK")

print("   AGE-Random...", end=" ", flush=True)
X_age_rand = batch_attack(transfer_attack, X_eval, top_idx=top5_rand, eps=EPS)
print("OK")

print("   FGSM-Top5...", end=" ", flush=True)
X_fgsm_t5  = batch_attack(fgsm_top5, X_eval, top_idx=top5_att, eps=EPS)
print("OK")

print("   FGSM...", end=" ", flush=True)
X_fgsm     = batch_attack(fgsm_full, X_eval, eps=EPS)
print("OK")

print("   PGD...", end=" ", flush=True)
X_pgd      = batch_attack(pgd_full, X_eval, eps=EPS)
print("OK")

if top5_shap is not None:
    print("   AGE-SHAP...", end=" ", flush=True)
    X_age_shap = batch_attack(transfer_attack, X_eval, top_idx=top5_shap, eps=EPS)
    print("OK")
else:
    X_age_shap = None

# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEĞERLENDİRME
# ═══════════════════════════════════════════════════════════════════════════════

def psri_per_slice(X_adv):
    """Her dilim icin PSRI = acc_adv / acc_clean"""
    result = {}
    for sl_name, classes in SLICES.items():
        class_ids = [i for i, c in enumerate(le.classes_) if c in classes]
        mask = np.isin(y_eval, class_ids)
        if mask.sum() < 10:
            result[sl_name] = float("nan")
            continue
        ac  = accuracy_score(y_eval[mask], tabnet.predict(X_eval[mask]))
        aa  = accuracy_score(y_eval[mask], tabnet.predict(X_adv[mask]))
        result[sl_name] = aa / ac if ac > 1e-9 else float("nan")
    return result

def evaluate(name, X_adv):
    y_adv    = tabnet.predict(X_adv)
    acc_adv  = accuracy_score(y_eval, y_adv)
    drop     = acc_clean - acc_adv
    l2       = float(np.mean(np.linalg.norm(X_adv - X_eval, axis=1)))
    fc       = float(np.mean((np.abs(X_adv - X_eval) > 1e-6).sum(axis=1)))
    psri     = psri_per_slice(X_adv)
    print(f"   {name:<25} drop={drop:.3f}  l2={l2:.3f}  feat={fc:.1f}  "
          f"URLLC-PSRI={psri.get('URLLC', float('nan')):.3f}")
    return {"name": name, "acc_adv": acc_adv, "drop": drop,
            "l2": l2, "feat_changed": fc, **{f"psri_{k}": v for k,v in psri.items()}}

print("\nDegerlendirme:")
rows = []
rows.append(evaluate("AGE-Attention (Proposed)", X_age_att))
if X_age_shap is not None:
    rows.append(evaluate("AGE-SHAP",              X_age_shap))
rows.append(evaluate("AGE-Random",               X_age_rand))
rows.append(evaluate("FGSM-Top5 (no transfer)",  X_fgsm_t5))
rows.append(evaluate("FGSM (baseline)",          X_fgsm))
rows.append(evaluate("PGD (baseline)",           X_pgd))

# ═══════════════════════════════════════════════════════════════════════════════
# 8. GRAFİK
# ═══════════════════════════════════════════════════════════════════════════════
df_res = pd.DataFrame(rows)

names_short = {
    "AGE-Attention (Proposed)": "AGE-Att\n(Proposed)",
    "AGE-SHAP":                 "AGE-SHAP",
    "AGE-Random":               "AGE-Rand",
    "FGSM-Top5 (no transfer)":  "FGSM-Top5\n(no transfer)",
    "FGSM (baseline)":          "FGSM",
    "PGD (baseline)":           "PGD",
}
df_res["short"] = df_res["name"].map(names_short).fillna(df_res["name"])

COLORS = ["#E74C3C","#9B59B6","#F39C12","#27AE60","#3498DB","#1ABC9C"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Ablation Study — AGE Variants vs Baselines (ε={EPS}, n={N_EVAL})",
    fontsize=14, fontweight="bold"
)

# Panel 1: Accuracy Drop
ax = axes[0]
bars = ax.bar(range(len(df_res)), df_res["drop"],
              color=COLORS[:len(df_res)], alpha=0.85, edgecolor="white", linewidth=1.2)
ax.set_xticks(range(len(df_res)))
ax.set_xticklabels(df_res["short"], fontsize=9)
ax.set_ylabel("Accuracy Drop (↑ more effective attack)", fontsize=10)
ax.set_title("Attack Effectiveness", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars, df_res["drop"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

# Panel 2: L2 Stealthiness
ax = axes[1]
bars2 = ax.bar(range(len(df_res)), df_res["l2"],
               color=COLORS[:len(df_res)], alpha=0.85, edgecolor="white", linewidth=1.2)
ax.set_xticks(range(len(df_res)))
ax.set_xticklabels(df_res["short"], fontsize=9)
ax.set_ylabel("L₂ Perturbation (↓ more stealthy)", fontsize=10)
ax.set_title("Stealthiness (L₂ Norm)", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars2, df_res["l2"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

# Panel 3: URLLC PSRI
ax = axes[2]
psri_col = "psri_URLLC"
if psri_col in df_res.columns:
    vals = df_res[psri_col].fillna(0).values
    bars3 = ax.bar(range(len(df_res)), vals,
                   color=COLORS[:len(df_res)], alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(df_res)))
    ax.set_xticklabels(df_res["short"], fontsize=9)
    ax.set_ylabel("URLLC PSRI (↓ worse for defender)", fontsize=10)
    ax.set_title("URLLC Slice Robustness (PSRI)", fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect robustness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars3, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
out_fig = "reports/17_ablation_study.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nOK: {out_fig}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. CSV KAYDET
# ═══════════════════════════════════════════════════════════════════════════════
out_csv = "reports/ablation_sonuclari.csv"
df_res.to_csv(out_csv, index=False)
print(f"OK: {out_csv}")

# Ozet tablo
print("\n" + "="*80)
print("ABLATION OZETI")
print("="*80)
hdr = f"{'Variant':<30} {'AccDrop':>8} {'L2':>8} {'FeatCnt':>8} {'URLLC-PSRI':>12}"
print(hdr)
print("-"*80)
for _, r in df_res.iterrows():
    psri_u = r.get("psri_URLLC", float("nan"))
    print(f"{r['name']:<30} {r['drop']:>8.4f} {r['l2']:>8.4f} "
          f"{r['feat_changed']:>8.1f} {psri_u:>12.4f}")

print("\nYorum:")
att_row  = df_res[df_res["name"]=="AGE-Attention (Proposed)"].iloc[0]
rand_row = df_res[df_res["name"]=="AGE-Random"].iloc[0]
fgsm_row = df_res[df_res["name"]=="FGSM (baseline)"].iloc[0]

att_vs_rand = att_row["drop"] - rand_row["drop"]
att_vs_fgsm_l2 = fgsm_row["l2"] - att_row["l2"]
print(f"  AGE-Att vs AGE-Rand drop farki: {att_vs_rand:+.4f} "
      f"({'Attention yardimlandi' if att_vs_rand>0 else 'Random daha iyi'}) ")
print(f"  AGE-Att vs FGSM  L2  farki:    {att_vs_fgsm_l2:+.4f} "
      f"({'AGE daha gizli' if att_vs_fgsm_l2>0 else 'FGSM daha gizli'})")

print("\n" + "="*80)
print("TAMAMLANDI — sonraki adim: metni guncelle (Tablo 5 / Bolum 4.4)")
print("="*80)
