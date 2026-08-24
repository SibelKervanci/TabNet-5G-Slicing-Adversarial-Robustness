"""
09_toniot_generalizability.py
TON_IoT Network Dataset — Generalizability Analysis
TabNet + AGE (Transfer) + PSRI on TON_IoT

Çalıştırma: python 09_toniot_generalizability.py
"""

import os, warnings, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier as TabNet
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OK: PyTorch {torch.__version__} | {DEVICE}")
except ImportError:
    print("HATA: pip install pytorch-tabnet"); exit(1)

# ── 5G DİLİM HARİTASI ────────────────────────────────────────────────────────
SLICE_MAP = {
    "eMBB":   ["ddos"],
    "URLLC":  ["dos", "mitm"],
    "mMTC":   ["backdoor", "injection", "xss", "scanning", "password", "ransomware"],
    "Benign": ["normal"],
}
CLASS_TO_SLICE = {cls: sl for sl, classes in SLICE_MAP.items() for cls in classes}

# ── 1. VERİ YÜKLE ────────────────────────────────────────────────────────────
print("\nTON_IoT verisi yukleniyor...")

BASE = os.path.join("CICIOT2023", "train")
data_path = os.path.join(BASE, "train_test_network.csv")

df = pd.read_csv(data_path, low_memory=False)

label_col = "type"

# Kategorik sutunlari encode et
cat_encode = ["proto", "service", "conn_state"]
for col in cat_encode:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Kategorik sütunları çıkar
exclude = ["src_ip","dst_ip",
           "dns_query","dns_qclass","dns_qtype","dns_rcode",
           "ssl_version","ssl_cipher","ssl_resumed","ssl_established",
           "ssl_subject","ssl_issuer","http_method","http_uri",
           "http_version","http_user_agent","http_orig_mime_types",
           "http_resp_mime_types","weird_name","weird_addl","weird_notice",
           "label", label_col]

feature_cols = [c for c in df.columns
                if c not in exclude and
                pd.api.types.is_numeric_dtype(df[c])]

print(f"OK: {len(feature_cols)} ozellik (sayisal + encode edilmis kategorik)")

# Train/test split
X_all = df[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0).values
y_all = df[label_col].str.lower().values

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

le = LabelEncoder().fit(y_all)
y_train = le.transform(y_train_r)
y_test  = le.transform(y_test_r)
class_names = le.classes_

scaler  = StandardScaler().fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test  = scaler.transform(X_test_r)

slice_labels = np.array([CLASS_TO_SLICE.get(c, "Unknown") for c in y_test_r])

print(f"OK: Train={len(X_train):,} | Test={len(X_test):,}")
print("Dilim dagilimi (test):")
for sl in ["eMBB","URLLC","mMTC","Benign"]:
    cnt = (slice_labels == sl).sum()
    print(f"   {sl:<8} {cnt:>6,}  ({cnt/len(X_test)*100:.1f}%)")

# ── 2. TABNET EĞİTİMİ ────────────────────────────────────────────────────────
print("\nTabNet egitiliyor (TON_IoT)...")
t0 = time.time()

model = TabNet(
    n_d=32, n_a=32, n_steps=5,
    gamma=1.3, n_independent=2, n_shared=2,
    momentum=0.02,
    optimizer_fn=torch.optim.Adam,
    device_name=DEVICE, seed=42, verbose=0
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test[:5000], y_test[:5000])],
    eval_metric=["accuracy"],
    batch_size=4096,
    max_epochs=50,
    patience=10,
)
elapsed = time.time() - t0

y_pred_clean = model.predict(X_test)
acc_clean    = accuracy_score(y_test, y_pred_clean)
f1_clean     = f1_score(y_test, y_pred_clean, average="weighted", zero_division=0)
f1_mac_clean = f1_score(y_test, y_pred_clean, average="macro", zero_division=0)

print(f"OK: TabNet TON_IoT — Acc={acc_clean:.4f} F1_w={f1_clean:.4f} F1_mac={f1_mac_clean:.4f} ({elapsed:.0f}s)")

# ── 3. ATTENTION MASK ────────────────────────────────────────────────────────
print("\nAttention mask hesaplaniyor...")
N_ATT = min(2000, len(X_test))
idx_att = np.random.choice(len(X_test), N_ATT, replace=False)
X_att   = X_test[idx_att]

explain_matrix, masks = model.explain(X_att)
feat_imp = pd.Series(
    explain_matrix.mean(axis=0),
    index=feature_cols
).sort_values(ascending=False)

print("Top-10 attention ozellikleri:")
for i, (feat, val) in enumerate(feat_imp.head(10).items(), 1):
    print(f"  {i:2}. {feat:<30} {val:.4f}")

top_feat_idx = list(range(min(5, len(feature_cols))))
for i, feat in enumerate(feat_imp.head(5).index):
    if feat in feature_cols:
        top_feat_idx[i] = feature_cols.index(feat)

att_mask = np.zeros(len(feature_cols))
for i, idx in enumerate(top_feat_idx):
    att_mask[idx] = feat_imp.iloc[i]
if att_mask.sum() > 0:
    att_mask = att_mask / att_mask.sum()
att_tensor = torch.FloatTensor(att_mask).to(DEVICE)

# ── 4. SURROGATE MLP ─────────────────────────────────────────────────────────
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
            if i < len(sklearn_mlp.coefs_) - 1:
                layers.append(nn.ReLU())
            in_size = out_size
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

torch_surr = SurrogateMLP(surrogate).to(DEVICE)
torch_surr.eval()

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
print("\nPSRI hesaplaniyor (epsilon=0.10)...")

N_EVAL = min(8000, len(X_test))
idx    = np.random.choice(len(X_test), N_EVAL, replace=False)
X_eval     = X_test[idx]
y_eval     = y_test[idx]
y_str_eval = y_test_r[idx]
sl_eval    = slice_labels[idx]

X_age  = transfer_age(X_eval, y_eval, epsilon=0.10)
X_fgsm = fgsm_transfer(X_eval, y_eval, epsilon=0.10)

y_clean_p = model.predict(X_eval)
y_age_p   = model.predict(X_age)
y_fgsm_p  = model.predict(X_fgsm)

l2_age  = l2_norm(X_eval, X_age)
l2_fgsm = l2_norm(X_eval, X_fgsm)
fc_age  = feat_changed(X_eval, X_age)
fc_fgsm = feat_changed(X_eval, X_fgsm)

print(f"\nGizlilik (ε=0.10):")
print(f"  AGE  L2={l2_age:.3f}  Feat={fc_age:.1f}/{len(feature_cols)}")
print(f"  FGSM L2={l2_fgsm:.3f}  Feat={fc_fgsm:.1f}/{len(feature_cols)}")

results = {}
slices  = ["eMBB","URLLC","mMTC","Benign","GLOBAL"]

for sl in slices:
    mask = np.ones(len(y_eval), dtype=bool) if sl == "GLOBAL" else (sl_eval == sl)
    if mask.sum() == 0:
        continue
    yc = y_clean_p[mask]; ya = y_age_p[mask]; yf = y_fgsm_p[mask]; yt = y_eval[mask]
    acc_c  = accuracy_score(yt, yc)
    acc_a  = accuracy_score(yt, ya)
    acc_f  = accuracy_score(yt, yf)
    f1_c   = f1_score(yt, yc, average="weighted", zero_division=0)
    psri_a = acc_a / (acc_c + 1e-9)
    psri_f = acc_f / (acc_c + 1e-9)
    results[sl] = {
        "n": mask.sum(),
        "acc_clean": acc_c, "acc_age": acc_a, "acc_fgsm": acc_f,
        "f1_clean": f1_c,
        "psri_age": psri_a, "psri_fgsm": psri_f,
    }

# ── 7. RAPOR ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("TON_IoT PSRI SONUÇLARI (ε=0.10)")
print("="*65)
print(f"\n{'Dilim':<10} {'N':>6} {'Temiz':>8} {'AGE':>8} {'FGSM':>8} {'PSRI(AGE)':>10} {'PSRI(FGSM)':>12}")
print("-"*65)
for sl in slices:
    if sl not in results:
        continue
    r = results[sl]
    print(f"{sl:<10} {r['n']:>6,} {r['acc_clean']:>8.3f} {r['acc_age']:>8.3f} "
          f"{r['acc_fgsm']:>8.3f} {r['psri_age']:>10.3f} {r['psri_fgsm']:>12.3f}")

# ── 8. GRAFİK ────────────────────────────────────────────────────────────────
slice_order = [s for s in slices if s in results]
psri_age_v  = [results[s]["psri_age"]  for s in slice_order]
psri_fgsm_v = [results[s]["psri_fgsm"] for s in slice_order]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x = np.arange(len(slice_order))
w = 0.35
bars1 = ax.bar(x-w/2, psri_age_v,  w, label="AGE (Transfer)", color="#E74C3C", alpha=0.85)
bars2 = ax.bar(x+w/2, psri_fgsm_v, w, label="FGSM",           color="#3498DB", alpha=0.85)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect robustness (1.0)")
ax.set_xticks(x); ax.set_xticklabels(slice_order, fontsize=10)
ax.set_ylabel("PSRI (higher = more robust)"); ax.set_ylim(0, 1.15)
ax.set_title("TON_IoT — Per-Slice PSRI", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=8, color="#E74C3C")
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=8, color="#3498DB")

ax = axes[1]
ciciot_psri_age  = [0.488, 0.023, 0.983, 1.000, 0.444]
ciciot_psri_fgsm = [0.603, 0.022, 0.453, 1.006, 0.502]

x2 = np.arange(len(slice_order))
w2 = 0.2
ax.bar(x2-w2*1.5, ciciot_psri_age[:len(slice_order)],  w2, label="CICIoT — AGE",  color="#E74C3C", alpha=0.85)
ax.bar(x2-w2*0.5, ciciot_psri_fgsm[:len(slice_order)], w2, label="CICIoT — FGSM", color="#3498DB", alpha=0.85)
ax.bar(x2+w2*0.5, psri_age_v,  w2, label="TON_IoT — AGE",  color="#E74C3C", alpha=0.4, hatch="//")
ax.bar(x2+w2*1.5, psri_fgsm_v, w2, label="TON_IoT — FGSM", color="#3498DB", alpha=0.4, hatch="//")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xticks(x2); ax.set_xticklabels(slice_order, fontsize=10)
ax.set_ylabel("PSRI"); ax.set_ylim(0, 1.2)
ax.set_title("CICIoT2023 vs TON_IoT — PSRI Comparison", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("reports/13_toniot_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: reports/13_toniot_comparison.png")

pd.DataFrame([
    {"Dataset": "TON_IoT", "Dilim": sl,
     "N": results[sl]["n"],
     "Acc_Clean": round(results[sl]["acc_clean"], 4),
     "Acc_AGE":   round(results[sl]["acc_age"],   4),
     "Acc_FGSM":  round(results[sl]["acc_fgsm"],  4),
     "F1_Clean":  round(results[sl]["f1_clean"],  4),
     "PSRI_AGE":  round(results[sl]["psri_age"],  4),
     "PSRI_FGSM": round(results[sl]["psri_fgsm"], 4),
     "L2_AGE":    round(l2_age, 4),
     "L2_FGSM":   round(l2_fgsm, 4),
     "Feat_AGE":  round(fc_age, 1),
     "Feat_FGSM": round(fc_fgsm, 1),
    }
    for sl in slice_order if sl in results
]).to_csv("reports/toniot_psri_sonuclari.csv", index=False)
print("OK: reports/toniot_psri_sonuclari.csv")

print("\n" + "="*65)
print("TAMAMLANDI — Sonuclari paylasin")
print("="*65)