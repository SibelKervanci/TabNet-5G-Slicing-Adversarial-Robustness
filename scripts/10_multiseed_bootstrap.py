hocam dosyanın içini bunla değiş :"""
10_multiseed_bootstrap.py
1. TabNet çoklu seed (5 seed) — PSRI mean±std
2. Bootstrap CI: AGE vs FGSM accuracy drop

Çalıştırma: python 10_multiseed_bootstrap.py
"""

import os, warnings, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
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
print(f"OK: Train={len(X_train):,} | Test={len(X_test):,} | Val={len(X_val):,}")

# ── ATTENTION MASK (validation setinden) ─────────────────────────────────────
print("\nAttention mask yukleniyor (validation set)...")
base_model = TabNetClassifier()
base_model.load_model("models/tabnet_ciciot2023.zip")

N_ATT = 2000
idx_att = np.random.choice(len(X_val), N_ATT, replace=False)
explain_matrix, _ = base_model.explain(X_val[idx_att])
feat_imp = pd.Series(explain_matrix.mean(axis=0), index=feature_cols).sort_values(ascending=False)
top5_attention_idx = [feature_cols.index(f) for f in feat_imp.head(5).index]
att_weights = feat_imp.head(5).values.tolist()
print(f"OK: Top-5 attention: {feat_imp.head(5).index.tolist()}")

# ── SURROGATE MLP ─────────────────────────────────────────────────────────────
surrogate_base = MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
                               max_iter=30, random_state=42,
                               early_stopping=True, validation_fraction=0.1, verbose=False)
surrogate_base.fit(X_train, y_train)

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

torch_surr = SurrogateMLP(surrogate_base).to(DEVICE)
torch_surr.eval()

# ── SALDIRI FONKSİYONLARI ────────────────────────────────────────────────────
def make_att_tensor(feat_idx, weights=None):
    mask = np.zeros(len(feature_cols))
    for i, idx in enumerate(feat_idx):
        mask[idx] = weights[i] if weights is not None else 1.0
    if mask.sum() > 0:
        mask = mask / mask.sum()
    return torch.FloatTensor(mask).to(DEVICE)

def transfer_age(X, y, feat_idx, att_w=None, epsilon=0.10, n_steps=10):
    att_t = make_att_tensor(feat_idx, att_w)
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

def compute_psri(model, X_eval, y_eval, X_adv, sl_eval, slices):
    results = {}
    y_clean = model.predict(X_eval)
    y_adv   = model.predict(X_adv)
    for sl in slices:
        mask = np.ones(len(y_eval), dtype=bool) if sl == "GLOBAL" else (sl_eval == sl)
        if mask.sum() == 0:
            continue
        acc_c = accuracy_score(y_eval[mask], y_clean[mask])
        acc_a = accuracy_score(y_eval[mask], y_adv[mask])
        results[sl] = {"acc_clean": acc_c, "acc_adv": acc_a,
                       "psri": acc_a/(acc_c+1e-9), "n": mask.sum()}
    return results

# ── EVAL SET ─────────────────────────────────────────────────────────────────
N_EVAL  = 8000
idx_e   = np.random.choice(len(X_test), N_EVAL, replace=False)
X_eval  = X_test[idx_e]
y_eval  = y_test[idx_e]
sl_eval = slice_labels[idx_e]
SLICES  = ["eMBB","URLLC","mMTC","Benign","GLOBAL"]

print("\nSaldirilar uretiliyor...")
X_age_att = transfer_age(X_eval, y_eval, top5_attention_idx, att_weights, epsilon=0.10)
X_fgsm    = fgsm_transfer(X_eval, y_eval, epsilon=0.10)
print("OK: Saldirilar hazir")

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: TABNET ÇOKLU SEED
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("BÖLÜM 1: TabNet Çoklu Seed (5 seed)")
print("="*65)

SEEDS = [42, 123, 456, 789, 1000]
seed_psri = {sl: [] for sl in SLICES}
seed_acc  = []

for seed in SEEDS:
    print(f"\n  Seed {seed} egitiliyor...")
    model = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5,
        gamma=1.3, n_independent=2, n_shared=2,
        momentum=0.02, optimizer_fn=torch.optim.Adam,
        device_name=DEVICE, seed=seed, verbose=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val[:5000], y_val[:5000])],
        eval_metric=["accuracy"],
        batch_size=4096, virtual_batch_size=256,
        max_epochs=50, patience=10,
    )

    surr = MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
                         max_iter=30, random_state=seed,
                         early_stopping=True, validation_fraction=0.1, verbose=False)
    surr.fit(X_train, y_train)
    torch_s = SurrogateMLP(surr).to(DEVICE); torch_s.eval()

    def age_seed(X, y, feat_idx, weights, epsilon=0.10, n_steps=10):
        att_t = make_att_tensor(feat_idx, weights)
        step  = epsilon / n_steps
        X_t   = torch.FloatTensor(X).to(DEVICE)
        y_t   = torch.LongTensor(y).to(DEVICE)
        X_adv = X_t.clone().detach()
        for _ in range(n_steps):
            X_adv.requires_grad_(True)
            out  = torch_s(X_adv)
            loss = F.cross_entropy(out, y_t)
            loss.backward()
            with torch.no_grad():
                pert  = step * X_adv.grad.sign() * att_t.unsqueeze(0)
                X_adv = X_adv + pert
                delta = torch.clamp(X_adv - X_t, -epsilon, epsilon)
                X_adv = (X_t + delta).detach()
        return X_adv.cpu().numpy()

    X_age_s = age_seed(X_eval, y_eval, top5_attention_idx, att_weights)
    acc = accuracy_score(y_eval, model.predict(X_eval))
    seed_acc.append(acc)
    print(f"  Acc={acc:.4f}")

    psri = compute_psri(model, X_eval, y_eval, X_age_s, sl_eval, SLICES)
    for sl in SLICES:
        if sl in psri:
            seed_psri[sl].append(psri[sl]["psri"])

print("\n" + "="*65)
print("TabNet Multi-Seed PSRI Sonuçlari (AGE, ε=0.10)")
print("="*65)
print(f"\n{'Dilim':<10} {'PSRI mean':>12} {'PSRI std':>10} {'95% CI':>18}")
print("-"*55)
multiseed_results = {}
for sl in SLICES:
    if sl in seed_psri and len(seed_psri[sl]) > 0:
        vals  = seed_psri[sl]
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        ci_lo  = mean_v - 1.96*std_v/np.sqrt(len(vals))
        ci_hi  = mean_v + 1.96*std_v/np.sqrt(len(vals))
        multiseed_results[sl] = {"mean": mean_v, "std": std_v, "ci_lo": ci_lo, "ci_hi": ci_hi}
        print(f"{sl:<10} {mean_v:>12.3f} {std_v:>10.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

print(f"\nTabNet Accuracy: {np.mean(seed_acc):.4f} ± {np.std(seed_acc):.4f}")

# ════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: BOOTSTRAP CI
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("BÖLÜM 2: Bootstrap Güven Aralığı (AGE vs FGSM)")
print("="*65)

eval_model = TabNetClassifier()
eval_model.load_model("models/tabnet_ciciot2023.zip")

y_clean = eval_model.predict(X_eval)
y_age   = eval_model.predict(X_age_att)
y_fgsm_ = eval_model.predict(X_fgsm)

def bootstrap_ci(y_true, y_pred1, y_pred2, n_boot=1000, ci=95):
    n = len(y_true)
    diffs = []
    base_acc = accuracy_score(y_true, y_clean)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        drop1 = base_acc - accuracy_score(y_true[idx], y_pred1[idx])
        drop2 = base_acc - accuracy_score(y_true[idx], y_pred2[idx])
        diffs.append(drop1 - drop2)
    lo  = np.percentile(diffs, (100-ci)/2)
    hi  = np.percentile(diffs, 100-(100-ci)/2)
    obs = (base_acc - accuracy_score(y_true, y_pred1)) - (base_acc - accuracy_score(y_true, y_pred2))
    return obs, lo, hi

print("\nHesaplaniyor (1000 bootstrap)...")
obs, lo, hi = bootstrap_ci(y_eval, y_age, y_fgsm_)
print(f"\nAGE drop - FGSM drop:")
print(f"  Observed difference: {obs:+.4f}")
print(f"  95% Bootstrap CI:    [{lo:.4f}, {hi:.4f}]")
if lo > 0:
    print("  Sonuc: AGE istatistiksel olarak daha etkili (CI tamamen pozitif)")
elif hi < 0:
    print("  Sonuc: FGSM istatistiksel olarak daha etkili (CI tamamen negatif)")
else:
    print("  Sonuc: Fark istatistiksel olarak ANLAMSIZ (CI sifiri iceriyor)")

# ── GRAFİK ───────────────────────────────────────────────────────────────────
sl_list = [s for s in SLICES if s in multiseed_results]
means   = [multiseed_results[s]["mean"] for s in sl_list]
stds    = [multiseed_results[s]["std"]  for s in sl_list]
x = np.arange(len(sl_list))

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(x, means, color=["#2E5090","#E74C3C","#3B6D11","#888888","#534AB7"], alpha=0.85)
ax.errorbar(x, means, yerr=[1.96*s/np.sqrt(5) for s in stds],
            fmt="none", color="black", capsize=5, lw=1.5)
ax.set_xticks(x); ax.set_xticklabels(sl_list, fontsize=10)
ax.set_ylabel("PSRI (AGE, ε=0.10)")
ax.set_title("TabNet PSRI — 5 Seeds (mean ± 95% CI)", fontweight="bold")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylim(0, 1.2); ax.grid(True, alpha=0.3, axis="y")
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("reports/14_multiseed_psri.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/14_multiseed_psri.png")

# CSV
pd.DataFrame([
    {"Dilim": sl,
     "PSRI_mean": round(multiseed_results[sl]["mean"], 4),
     "PSRI_std":  round(multiseed_results[sl]["std"],  4),
     "CI_lo":     round(multiseed_results[sl]["ci_lo"],4),
     "CI_hi":     round(multiseed_results[sl]["ci_hi"],4)}
    for sl in sl_list
]).to_csv("reports/multiseed_psri.csv", index=False)
print("OK: reports/multiseed_psri.csv")
print(f"\nBootstrap CI: obs={obs:+.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")
print("\n" + "="*65)
print("TAMAMLANDI")
print("="*65)