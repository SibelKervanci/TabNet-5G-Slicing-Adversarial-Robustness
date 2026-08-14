"""
04c_age_transfer.py
Transfer tabanlı AGE saldırısı — Surrogate MLP üzerinden TabNet'e transfer.

Fikir:
1. MLP surrogate model eğit (gradient akışı düzgün)
2. AGE'yi surrogate üzerinde uygula → adversarial örnekler üret
3. Bu örnekleri TabNet'e ver → transfer başarısı ölç
4. FGSM ve PGD ile karşılaştır

Çalıştırma: python 04c_age_transfer.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
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

# ── 1. VERİ ──────────────────────────────────────────────────────────────────
print("\nVeri yukleniyor...")
df_train = pd.read_csv(os.path.join("CICIOT2023", "train", "train.csv"),
                       low_memory=False, nrows=100000)
df_test  = pd.read_csv(os.path.join("CICIOT2023", "test",  "test.csv"),
                       low_memory=False, nrows=20000)

label_col    = "label"
feature_cols = [c for c in df_train.columns
                if c != label_col and df_train[c].dtype in [np.float64, np.int64, float, int]]

def prep(df):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df[label_col].values
    return X, y

X_train_r, y_train_r = prep(df_train)
X_test_r,  y_test_r  = prep(df_test)

le = LabelEncoder().fit(np.concatenate([y_train_r, y_test_r]))
y_train = le.transform(y_train_r)
y_test  = le.transform(y_test_r)

scaler  = StandardScaler().fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test  = scaler.transform(X_test_r)

print(f"OK: Train={len(X_train):,} | Test={len(X_test):,} | {len(feature_cols)} ozellik")

# ── 2. TABNET YUKLE ───────────────────────────────────────────────────────────
print("\nTabNet yukleniyor...")
tabnet = TabNetClassifier()
tabnet.load_model("models/tabnet_ciciot2023.zip")
tabnet.network.eval()

y_pred_clean = tabnet.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_clean)
print(f"OK: TabNet baseline accuracy: {baseline_acc:.4f}")

# ── 3. SURROGATE MLP EĞİTİMİ ─────────────────────────────────────────────────
print("\nSurrogate MLP egitiliyor...")
surrogate = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    max_iter=50,
    random_state=42,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.1
)
surrogate.fit(X_train, y_train)
surr_acc = accuracy_score(y_test, surrogate.predict(X_test))
print(f"OK: Surrogate MLP accuracy: {surr_acc:.4f}")

# ── 4. PYTORCH SURROGATE (gradient için) ──────────────────────────────────────
# Sklearn MLP'nin ağırlıklarını PyTorch'a aktarıyoruz
class SurrogateMLP(nn.Module):
    def __init__(self, sklearn_mlp, n_features, n_classes):
        super().__init__()
        layers = []
        in_size = n_features
        for coef, intercept in zip(sklearn_mlp.coefs_, sklearn_mlp.intercepts_):
            out_size = coef.shape[1]
            layer = nn.Linear(in_size, out_size)
            layer.weight = nn.Parameter(torch.FloatTensor(coef.T))
            layer.bias   = nn.Parameter(torch.FloatTensor(intercept))
            layers.append(layer)
            if len(layers) < len(sklearn_mlp.coefs_):
                layers.append(nn.ReLU())
            in_size = out_size
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

n_classes  = len(le.classes_)
n_features = X_train.shape[1]
torch_surr = SurrogateMLP(surrogate, n_features, n_classes).to(DEVICE)
torch_surr.eval()
print("OK: PyTorch surrogate hazir")

# ── 5. ATTENTION OZELLIKLERI ─────────────────────────────────────────────────
top_feats = []
if os.path.exists("reports/top_attention_features.csv"):
    top_feats    = pd.read_csv("reports/top_attention_features.csv",
                               header=None)[0].tolist()
    top_feat_idx = [feature_cols.index(f) for f in top_feats if f in feature_cols]
    print(f"OK: AGE hedefleri: {top_feats}")
else:
    top_feat_idx = list(range(5))
    print("UYARI: top_attention_features.csv bulunamadi, ilk 5 ozellik kullaniliyor")

# ── 6. SALDIRI FONKSİYONLARI ─────────────────────────────────────────────────
def transfer_age(X, y, epsilon, n_steps=10, feat_idx=None):
    """
    Transfer tabanlı AGE:
    Surrogate MLP üzerinde attention-guided gradient saldırısı uygula,
    üretilen adversarial örnekleri TabNet'e transfer et.
    """
    if feat_idx is None:
        feat_idx = list(range(X.shape[1]))

    step_size = epsilon / n_steps
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y).to(DEVICE)

    # Attention maskesi — hangi özelliklere ne kadar pertürbasyon
    att_mask = np.zeros(X.shape[1])
    if os.path.exists("reports/full_feature_importance.csv"):
        fi = pd.read_csv("reports/full_feature_importance.csv", index_col=0).squeeze()
        for idx in feat_idx:
            fname = feature_cols[idx]
            if fname in fi.index:
                att_mask[idx] = fi[fname]
    else:
        for idx in feat_idx:
            att_mask[idx] = 1.0

    if att_mask.sum() > 0:
        att_mask = att_mask / att_mask.sum()
    att_t = torch.FloatTensor(att_mask).to(DEVICE)

    X_adv = X_t.clone().detach()

    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        output = torch_surr(X_adv)
        loss   = F.cross_entropy(output, y_t)
        loss.backward()
        with torch.no_grad():
            grad  = X_adv.grad.data
            pert  = step_size * grad.sign() * att_t.unsqueeze(0)
            X_adv = X_adv + pert
            delta = torch.clamp(X_adv - X_t, -epsilon, epsilon)
            X_adv = (X_t + delta).detach()

    return X_adv.cpu().numpy()

def fgsm(X, y, epsilon):
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y).to(DEVICE)
    X_t.requires_grad_(True)
    output = torch_surr(X_t)
    loss   = F.cross_entropy(output, y_t)
    loss.backward()
    return (X_t + epsilon * X_t.grad.sign()).detach().cpu().numpy()

def pgd(X, y, epsilon, n_steps=10):
    step = epsilon / n_steps
    X_t  = torch.FloatTensor(X).to(DEVICE)
    y_t  = torch.LongTensor(y).to(DEVICE)
    X_adv = X_t.clone().detach()
    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        output = torch_surr(X_adv)
        loss   = F.cross_entropy(output, y_t)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + step * X_adv.grad.sign()
            delta = torch.clamp(X_adv - X_t, -epsilon, epsilon)
            X_adv = (X_t + delta).detach()
    return X_adv.cpu().numpy()

def l2_norm(X_orig, X_adv):
    return float(np.mean(np.linalg.norm(X_adv - X_orig, axis=1)))

def feat_changed(X_orig, X_adv, thr=1e-6):
    return float(np.mean((np.abs(X_adv - X_orig) > thr).sum(axis=1)))

# ── 7. EPSİLON TARAMA ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("TRANSFER TABANLI AGE — EPSİLON TARAMA")
print("Surrogate MLP → TabNet transfer")
print("="*65)

# Küçük örnek al
N_EVAL = min(2000, len(X_test))
idx    = np.random.choice(len(X_test), N_EVAL, replace=False)
X_eval = X_test[idx]
y_eval = y_test[idx]

epsilons = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
results  = []

for eps in epsilons:
    print(f"\n  ε={eps:.2f} hesaplaniyor...")

    X_age_t = transfer_age(X_eval, y_eval, epsilon=eps,
                           n_steps=10, feat_idx=top_feat_idx)
    X_fgsm  = fgsm(X_eval, y_eval, epsilon=eps)
    X_pgd   = pgd(X_eval,  y_eval, epsilon=eps, n_steps=10)

    # TabNet üzerinde test
    acc_age  = accuracy_score(y_eval, tabnet.predict(X_age_t))
    acc_fgsm = accuracy_score(y_eval, tabnet.predict(X_fgsm))
    acc_pgd  = accuracy_score(y_eval, tabnet.predict(X_pgd))

    # Surrogate üzerinde test (transfer oranı kontrolü)
    acc_surr_age  = accuracy_score(y_eval, surrogate.predict(X_age_t))
    acc_surr_fgsm = accuracy_score(y_eval, surrogate.predict(X_fgsm))

    l2_age  = l2_norm(X_eval, X_age_t)
    l2_fgsm = l2_norm(X_eval, X_fgsm)
    fc_age  = feat_changed(X_eval, X_age_t)
    fc_fgsm = feat_changed(X_eval, X_fgsm)

    results.append((eps, acc_age, acc_fgsm, acc_pgd,
                    acc_surr_age, acc_surr_fgsm,
                    l2_age, l2_fgsm, fc_age, fc_fgsm))

    print(f"  TabNet  — AGE: {acc_age:.3f}  FGSM: {acc_fgsm:.3f}  PGD: {acc_pgd:.3f}")
    print(f"  Surrogate— AGE: {acc_surr_age:.3f}  FGSM: {acc_surr_fgsm:.3f}")
    print(f"  L2      — AGE: {l2_age:.3f}  FGSM: {l2_fgsm:.3f}")
    print(f"  Feat    — AGE: {fc_age:.1f}  FGSM: {fc_fgsm:.1f}")

# ── 8. ÖZET ──────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ÖZET — TRANSFER BAŞARISI")
print(f"{'ε':>6} {'AGE(TN)':>10} {'FGSM(TN)':>10} {'AGE(Sur)':>10} {'L2 AGE':>8} {'Feat AGE':>10}")
print("-"*60)
for r in results:
    eps, aa, fa, pa, asa, asf, al, fl, afc, ffc = r
    print(f"{eps:>6.2f} {aa:>10.3f} {fa:>10.3f} {asa:>10.3f} {al:>8.3f} {afc:>10.1f}")

# ── 9. GRAFİK ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epsilons_arr = [r[0] for r in results]
acc_age_arr  = [r[1] for r in results]
acc_fgsm_arr = [r[2] for r in results]
acc_pgd_arr  = [r[3] for r in results]
l2_age_arr   = [r[6] for r in results]
l2_fgsm_arr  = [r[7] for r in results]

ax = axes[0]
ax.axhline(y=baseline_acc, color="gray", linestyle="--",
           alpha=0.6, label=f"Baseline ({baseline_acc:.3f})")
ax.plot(epsilons_arr, acc_age_arr,  "r-o", lw=2.5, ms=8, label="AGE-Transfer (Proposed)")
ax.plot(epsilons_arr, acc_fgsm_arr, "b--s",lw=2,   ms=7, label="FGSM-Transfer")
ax.plot(epsilons_arr, acc_pgd_arr,  "g:^", lw=2,   ms=7, label="PGD-Transfer")
ax.set_xlabel("Perturbation Magnitude (ε)")
ax.set_ylabel("TabNet Accuracy")
ax.set_title("")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

ax = axes[1]
ax.plot(epsilons_arr, l2_age_arr,  "r-o", lw=2.5, ms=8, label="AGE-Transfer")
ax.plot(epsilons_arr, l2_fgsm_arr, "b--s",lw=2,   ms=7, label="FGSM-Transfer")
ax.set_xlabel("Perturbation Magnitude (ε)")
ax.set_ylabel("Average L2 Norm")
ax.set_title("")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/10_transfer_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/10_transfer_age.png")

# CSV kaydet
pd.DataFrame(results,
    columns=["epsilon","acc_age_tabnet","acc_fgsm_tabnet","acc_pgd_tabnet",
             "acc_age_surrogate","acc_fgsm_surrogate",
             "l2_age","l2_fgsm","feat_age","feat_fgsm"]
).to_csv("reports/transfer_age_sonuclari.csv", index=False)
print("OK: reports/transfer_age_sonuclari.csv")

print("\n" + "="*65)
print("TAMAMLANDI")
print("="*65)
