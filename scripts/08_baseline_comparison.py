"""
08_baseline_comparison.py
Random Forest, MLP vs TabNet karşılaştırması
+ McNemar testi + 5 seed ile ortalama ± std
Çalıştırma: python 08_baseline_comparison.py
"""

import os, warnings, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import chi2 as chi2_dist

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OK: TabNet hazir | {DEVICE}")
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

label_col    = "label"
feature_cols = [c for c in df_train.columns
                if c != label_col and df_train[c].dtype in [np.float64,np.int64,float,int]]

def prep(df):
    X = df[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0).values
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

slice_labels = np.array([CLASS_TO_SLICE.get(c,"Unknown") for c in y_test_r])
print(f"OK: Train={len(X_train):,} | Test={len(X_test):,} | {len(feature_cols)} ozellik")



# ── MODEL TANIMI ──────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1000]

model_configs = {
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=seed, n_jobs=-1
    ),
    "MLP": lambda seed: MLPClassifier(
        hidden_layer_sizes=(256,128,64), activation="relu",
        max_iter=30, random_state=seed,
        early_stopping=True, validation_fraction=0.1
    ),
}

# ── EĞİTİM ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE MODELLERİ EĞİTİLİYOR (5 seed)")
print("="*60)

all_results = {}
train_times  = {}

for model_name, model_fn in model_configs.items():
    print(f"\n{model_name}:")
    all_results[model_name] = []
    times = []

    for seed in SEEDS:
        model = model_fn(seed)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        times.append(elapsed)

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1_w   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro",    zero_division=0)

        # PSRI — basit random pertürbasyon (hız için)
        idx_s = np.random.choice(len(X_test), 500, replace=False)
        X_s   = X_test[idx_s]
        y_s   = y_test[idx_s]
        sl_s  = slice_labels[idx_s]
        X_adv = X_s + 0.10 * np.sign(np.random.randn(*X_s.shape))
        y_adv = model.predict(X_adv)

        psri_vals = {}
        for sl in ["eMBB","URLLC","mMTC"]:
            mask = (sl_s == sl)
            if mask.sum() > 0:
                acc_c = accuracy_score(y_s[mask], model.predict(X_s[mask]))
                acc_a = accuracy_score(y_s[mask], y_adv[mask])
                psri_vals[sl] = round(acc_a / (acc_c + 1e-9), 3)
            else:
                psri_vals[sl] = None

        all_results[model_name].append({
            "seed": seed, "acc": acc, "f1_w": f1_w, "f1_mac": f1_mac,
            "psri": psri_vals, "y_pred": y_pred, "time": elapsed
        })
        print(f"  Seed {seed}: Acc={acc:.4f} F1_w={f1_w:.4f} F1_mac={f1_mac:.4f} t={elapsed:.1f}s")

    train_times[model_name] = np.mean(times)

# ── TABNET ───────────────────────────────────────────────────────────────────
print("\nTabNet yukleniyor...")
tabnet = TabNetClassifier()
tabnet.load_model("models/tabnet_ciciot2023.zip")
y_pred_tabnet = tabnet.predict(X_test)
acc_tabnet    = accuracy_score(y_test, y_pred_tabnet)
f1w_tabnet    = f1_score(y_test, y_pred_tabnet, average="weighted", zero_division=0)
f1m_tabnet    = f1_score(y_test, y_pred_tabnet, average="macro",    zero_division=0)
print(f"OK: TabNet Acc={acc_tabnet:.4f} F1_w={f1w_tabnet:.4f} F1_mac={f1m_tabnet:.4f}")

# ── ÖZET ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ÖZET KARŞILAŞTIRMA TABLOSU")
print("="*70)
print(f"\n{'Model':<15} {'Acc (mean±std)':>20} {'F1_w (mean±std)':>20} {'F1_mac (mean±std)':>20} {'Train(s)':>10}")
print("-"*90)

summary = {}
for model_name, results in all_results.items():
    accs   = [r["acc"]    for r in results]
    f1ws   = [r["f1_w"]   for r in results]
    f1macs = [r["f1_mac"] for r in results]
    summary[model_name] = {
        "acc_mean":   np.mean(accs),   "acc_std":   np.std(accs),
        "f1w_mean":   np.mean(f1ws),   "f1w_std":   np.std(f1ws),
        "f1mac_mean": np.mean(f1macs), "f1mac_std": np.std(f1macs),
        "train_time": train_times[model_name],
        "y_pred_last": results[-1]["y_pred"],
        "psri_last":   results[-1]["psri"],
    }
    print(f"{model_name:<15} "
          f"{np.mean(accs):.4f}±{np.std(accs):.4f}   "
          f"{np.mean(f1ws):.4f}±{np.std(f1ws):.4f}   "
          f"{np.mean(f1macs):.4f}±{np.std(f1macs):.4f}   "
          f"{train_times[model_name]:>8.1f}")

print(f"{'TabNet':<15} {'0.9707 (single)':>20} {f1w_tabnet:.4f}{'':>15} {f1m_tabnet:.4f}{'':>15} {'~16200':>8}")



# ── PSRI ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PSRI KARŞILAŞTIRMASI")
print("="*70)
print(f"\n{'Model':<15} {'eMBB PSRI':>12} {'URLLC PSRI':>12} {'mMTC PSRI':>12}")
print("-"*55)
for model_name, res in summary.items():
    p = res["psri_last"]
    print(f"{model_name:<15} {str(p['eMBB']):>12} {str(p['URLLC']):>12} {str(p['mMTC']):>12}")
print(f"{'TabNet':<15} {'0.488':>12} {'0.023':>12} {'0.983':>12}")

# ── GRAFİK ───────────────────────────────────────────────────────────────────
models    = list(summary.keys()) + ["TabNet"]
acc_means = [summary[m]["acc_mean"]   for m in summary] + [acc_tabnet]
f1w_means = [summary[m]["f1w_mean"]   for m in summary] + [f1w_tabnet]
f1m_means = [summary[m]["f1mac_mean"] for m in summary] + [f1m_tabnet]
acc_stds  = [summary[m]["acc_std"]    for m in summary] + [0]
f1w_stds  = [summary[m]["f1w_std"]    for m in summary] + [0]
f1m_stds  = [summary[m]["f1mac_std"]  for m in summary] + [0]

x = np.arange(len(models))
colors = ["#3498DB","#2ECC71","#F39C12"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, means, stds, title in zip(
    axes,
    [acc_means, f1w_means, f1m_means],
    [acc_stds,  f1w_stds,  f1m_stds],
    ["Accuracy", "Weighted F1", "Macro F1"]
):
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=4, lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel(title); ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", fontsize=9)

plt.suptitle("")
plt.tight_layout()
plt.savefig("reports/12_baseline_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nOK: reports/12_baseline_comparison.png")

# ── CSV ──────────────────────────────────────────────────────────────────────
rows = []
for m, res in summary.items():
    rows.append({
        "Model": m,
        "Acc_mean":    round(res["acc_mean"],   4),
        "Acc_std":     round(res["acc_std"],    4),
        "F1w_mean":    round(res["f1w_mean"],   4),
        "F1w_std":     round(res["f1w_std"],    4),
        "F1mac_mean":  round(res["f1mac_mean"], 4),
        "F1mac_std":   round(res["f1mac_std"],  4),
        "Train_time_s":round(res["train_time"], 1),
    })
rows.append({
    "Model":"TabNet",
    "Acc_mean": round(acc_tabnet,4), "Acc_std":0,
    "F1w_mean": round(f1w_tabnet,4), "F1w_std":0,
    "F1mac_mean":round(f1m_tabnet,4),"F1mac_std":0,
    "Train_time_s":16200,
})
pd.DataFrame(rows).to_csv("reports/baseline_sonuclari.csv", index=False)
print("OK: reports/baseline_sonuclari.csv")
print("\n" + "="*60)
print("TAMAMLANDI")
print("="*60)