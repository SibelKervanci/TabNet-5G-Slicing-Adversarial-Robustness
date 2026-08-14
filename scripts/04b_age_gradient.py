"""
04b_age_gradient.py
Gradient tabanlı AGE (Attention-Guided Evasion) saldırısı.
TabNet'in kayıp fonksiyonuna göre gerçek gradyan hesaplar.

Çalıştırma: python 04b_age_gradient.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    import torch.nn.functional as F
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OK: PyTorch {torch.__version__} | {DEVICE}")
except ImportError:
    print("HATA: pip install pytorch-tabnet"); exit(1)

# ── 1. MODEL VE VERİ ─────────────────────────────────────────────────────────
print("\nModel yukleniyor...")
model = TabNetClassifier()
model.load_model("models/tabnet_ciciot2023.zip")
model.network.eval()
print("OK")

print("Veri yukleniyor...")
df = pd.read_csv(os.path.join("CICIOT2023", "test", "test.csv"),
                 low_memory=False, nrows=50000)

label_col    = "label"
feature_cols = [c for c in df.columns
                if c != label_col and df[c].dtype in [np.float64, np.int64, float, int]]

X_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
y_str = df[label_col].values

le      = LabelEncoder()
y_full  = le.fit_transform(y_str)
scaler  = StandardScaler()
X_full  = scaler.fit_transform(X_raw)

# Değerlendirme için 2000 örnek
idx    = np.random.choice(len(X_full), 2000, replace=False)
X_eval = X_full[idx]
y_eval = y_full[idx]

# AGE hedef özellik indeksleri
top_feats = []
if os.path.exists("reports/top_attention_features.csv"):
    top_feats    = pd.read_csv("reports/top_attention_features.csv", header=None)[0].tolist()
    top_feat_idx = [feature_cols.index(f) for f in top_feats if f in feature_cols]
else:
    top_feat_idx = list(range(5))

print(f"OK: {len(X_eval)} ornek | AGE hedefleri: {top_feats}")

# Baseline
y_pred_clean = model.predict(X_eval)
baseline_acc = accuracy_score(y_eval, y_pred_clean)
print(f"Baseline accuracy: {baseline_acc:.4f}")

# ── 2. GRADİYAN TABANLI AGE ──────────────────────────────────────────────────
def gradient_age(X, y, epsilon, n_steps=10, attention_feat_idx=None):
    """
    Gradient tabanlı AGE saldırısı.
    
    Fikir:
    1. Her örnek için kayıp fonksiyonunun gradyanını hesapla
    2. Gradyanı yalnızca attention'ın yüksek olduğu özelliklere uygula
    3. PGD gibi iteratif adımlarla ilerle
    
    Bu sayede:
    - Pertürbasyon YÖNÜ gradyandan (doğru yön)
    - Pertürbasyon KONUMU attention'dan (doğru özellik)
    """
    if attention_feat_idx is None:
        attention_feat_idx = list(range(X.shape[1]))

    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.LongTensor(y).to(DEVICE)

    # Attention maskesi: hangi özelliklere ne kadar pertürbasyon
    explain_matrix, _ = model.explain(X)
    attention_weights  = np.mean(np.abs(explain_matrix), axis=0)
    att_mask           = np.zeros(X.shape[1])
    for fi in attention_feat_idx:
        att_mask[fi] = attention_weights[fi]
    # Normalize
    if att_mask.sum() > 0:
        att_mask = att_mask / att_mask.sum()
    att_mask_tensor = torch.FloatTensor(att_mask).to(DEVICE)

    # PGD tarzı iteratif saldırı — ama sadece attention özelliklerine
    step_size = epsilon / n_steps
    X_adv     = X_tensor.clone().detach()

    for step in range(n_steps):
        X_adv.requires_grad_(True)

        # TabNet forward pass
        output = model.network(X_adv)[0]  # logits
        loss   = F.cross_entropy(output, y_tensor)
        loss.backward()

        with torch.no_grad():
            grad = X_adv.grad.data
            # Gradyan işaretine göre pertürbasyon — attention maskesiyle ağırlıklandır
            grad_sign    = grad.sign()
            perturbation = step_size * grad_sign * att_mask_tensor.unsqueeze(0)
            X_adv        = X_adv + perturbation
            # Epsilon topu içinde tut
            delta  = X_adv - X_tensor
            delta  = torch.clamp(delta, -epsilon, epsilon)
            X_adv  = (X_tensor + delta).detach()

    return X_adv.cpu().numpy()

def fgsm(X, y, epsilon):
    """Karşılaştırma için standart FGSM"""
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y).to(DEVICE)
    X_t.requires_grad_(True)
    output = model.network(X_t)[0]
    loss   = F.cross_entropy(output, y_t)
    loss.backward()
    return (X_t + epsilon * X_t.grad.sign()).detach().cpu().numpy()

def pgd(X, y, epsilon, n_steps=10):
    """Karşılaştırma için standart PGD"""
    step_size = epsilon / n_steps
    X_t   = torch.FloatTensor(X).to(DEVICE)
    y_t   = torch.LongTensor(y).to(DEVICE)
    X_adv = X_t.clone().detach()
    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        output = model.network(X_adv)[0]
        loss   = F.cross_entropy(output, y_t)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + step_size * X_adv.grad.sign()
            delta = torch.clamp(X_adv - X_t, -epsilon, epsilon)
            X_adv = (X_t + delta).detach()
    return X_adv.cpu().numpy()

def l2_norm(X_orig, X_adv):
    return float(np.mean(np.linalg.norm(X_adv - X_orig, axis=1)))

def feat_changed(X_orig, X_adv, thr=1e-6):
    return float(np.mean((np.abs(X_adv - X_orig) > thr).sum(axis=1)))

# ── 3. EPSİLON TARAMA ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("GRADİYAN TABANLI AGE — EPSİLON TARAMA")
print("="*65)

epsilons = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
results  = []

for eps in epsilons:
    print(f"\n  ε={eps:.2f} hesaplaniyor...")

    X_age_g = gradient_age(X_eval, y_eval, epsilon=eps,
                           n_steps=10, attention_feat_idx=top_feat_idx)
    X_fgsm  = fgsm(X_eval, y_eval, epsilon=eps)
    X_pgd   = pgd(X_eval,  y_eval, epsilon=eps, n_steps=10)

    acc_age  = accuracy_score(y_eval, model.predict(X_age_g))
    acc_fgsm = accuracy_score(y_eval, model.predict(X_fgsm))
    acc_pgd  = accuracy_score(y_eval, model.predict(X_pgd))

    l2_age  = l2_norm(X_eval, X_age_g)
    l2_fgsm = l2_norm(X_eval, X_fgsm)
    fc_age  = feat_changed(X_eval, X_age_g)
    fc_fgsm = feat_changed(X_eval, X_fgsm)

    results.append((eps, acc_age, acc_fgsm, acc_pgd, l2_age, l2_fgsm, fc_age, fc_fgsm))

    print(f"  Acc  : AGE={acc_age:.3f}  FGSM={acc_fgsm:.3f}  PGD={acc_pgd:.3f}")
    print(f"  L2   : AGE={l2_age:.3f}   FGSM={l2_fgsm:.3f}")
    print(f"  Feat : AGE={fc_age:.1f}   FGSM={fc_fgsm:.1f}")

# ── 4. ÖZET ──────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ÖZET TABLO")
print(f"{'ε':>6} {'AGE Acc':>9} {'FGSM Acc':>9} {'PGD Acc':>9} "
      f"{'AGE L2':>8} {'FGSM L2':>8} {'AGE Feat':>9} {'FGSM Feat':>10}")
print("-"*75)
for r in results:
    eps, aa, fa, pa, al, fl, afc, ffc = r
    print(f"{eps:>6.2f} {aa:>9.3f} {fa:>9.3f} {pa:>9.3f} "
          f"{al:>8.3f} {fl:>8.3f} {afc:>9.1f} {ffc:>10.1f}")

# Verimlilik (L2 başına doğruluk düşüşü)
print("\nVERİMLİLİK (Düşüş / L2 — yüksek = gizli ve etkili):")
print(f"{'ε':>6} {'AGE Verim':>12} {'FGSM Verim':>12} {'AGE Üstünlüğü':>15}")
print("-"*50)
for r in results:
    eps, aa, fa, pa, al, fl, afc, ffc = r
    age_eff  = (baseline_acc - aa) / (al  + 1e-9)
    fgsm_eff = (baseline_acc - fa) / (fl  + 1e-9)
    ratio    = age_eff / (fgsm_eff + 1e-9)
    print(f"{eps:>6.2f} {age_eff:>12.2f} {fgsm_eff:>12.2f} {ratio:>14.2f}x")

# ── 5. GRAFİK ────────────────────────────────────────────────────────────────
epsilons_arr = [r[0] for r in results]
acc_age_arr  = [r[1] for r in results]
acc_fgsm_arr = [r[2] for r in results]
acc_pgd_arr  = [r[3] for r in results]
l2_age_arr   = [r[4] for r in results]
l2_fgsm_arr  = [r[5] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Gradient Tabanlı AGE vs FGSM vs PGD", fontsize=13, fontweight="bold")

# Accuracy
ax = axes[0]
ax.axhline(y=baseline_acc, color="gray", linestyle="--", alpha=0.6, label=f"Baseline ({baseline_acc:.3f})")
ax.plot(epsilons_arr, acc_age_arr,  "r-o", lw=2.5, ms=8, label="AGE-Grad (Önerilen)")
ax.plot(epsilons_arr, acc_fgsm_arr, "b--s",lw=2,   ms=7, label="FGSM")
ax.plot(epsilons_arr, acc_pgd_arr,  "g:^", lw=2,   ms=7, label="PGD")
ax.set_xlabel("ε"); ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs ε\n(Düşük = Etkili Saldırı)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# L2 norm
ax = axes[1]
ax.plot(epsilons_arr, l2_age_arr,  "r-o", lw=2.5, ms=8, label="AGE-Grad")
ax.plot(epsilons_arr, l2_fgsm_arr, "b--s",lw=2,   ms=7, label="FGSM")
ax.set_xlabel("ε"); ax.set_ylabel("Ortalama L2 Norm")
ax.set_title("Pertürbasyon Büyüklüğü\n(Düşük = Daha Gizli)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Verimlilik
ax = axes[2]
age_eff_arr  = [(baseline_acc - r[1]) / (r[4] + 1e-9) for r in results]
fgsm_eff_arr = [(baseline_acc - r[2]) / (r[5] + 1e-9) for r in results]
ax.plot(epsilons_arr, age_eff_arr,  "r-o", lw=2.5, ms=8, label="AGE-Grad")
ax.plot(epsilons_arr, fgsm_eff_arr, "b--s",lw=2,   ms=7, label="FGSM")
ax.set_xlabel("ε"); ax.set_ylabel("Verimlilik (Düşüş / L2)")
ax.set_title("Saldırı Verimliliği\n(Yüksek = Gizli & Etkili)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/09_gradient_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/09_gradient_age.png")

# CSV kaydet
pd.DataFrame(results, columns=["epsilon","acc_age","acc_fgsm","acc_pgd",
                                "l2_age","l2_fgsm","feat_age","feat_fgsm"]
).to_csv("reports/gradient_age_sonuclari.csv", index=False)
print("OK: reports/gradient_age_sonuclari.csv")

print("\n" + "="*65)
print("TAMAMLANDI — Sonuçları paylaşın, birlikte değerlendirelim.")
print("="*65)
