"""
03_tabnet_baseline.py 10-01-2026
CICIoT2023 - TabNet Baseline + Attention Mask Analysis
Folder structure: CICIOT2023/train/train.csv etc.
Run           : python3 03_tabnet_baseline.py
Requirements  : pip install pytorch-tabnet scikit-learn
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OK: PyTorch {torch.__version__} | Device: {DEVICE}")
except ImportError:
    print("ERROR: pip install pytorch-tabnet")
    exit(1)

# -- 1. LOAD DATA -------------------------------------------------------
BASE = "CICIOT2023"

def load(split):
    path = os.path.join(BASE, split, f"{split}.csv")
    return pd.read_csv(path, low_memory=False)

print("\nLoading data...")
df_train = load("train")
df_test  = load("test")
df_val   = load("validation")
print(f"  Train: {len(df_train):,} | Test: {len(df_test):,} | Val: {len(df_val):,}")

# -- 2. LABEL COLUMN ----------------------------------------------------
meta_label = None
if os.path.exists("reports/meta.txt"):
    for line in open("reports/meta.txt"):
        if line.startswith("label_col="):
            meta_label = line.strip().split("=")[1]

if meta_label and meta_label in df_train.columns:
    label_col = meta_label
else:
    for c in ["label","Label","attack_type","class","Class","target","Attack_type"]:
        if c in df_train.columns:
            label_col = c
            break

print(f"  Label column: '{label_col}'")

# -- 3. PREPROCESSING ---------------------------------------------------
feature_cols = [c for c in df_train.columns
                if c != label_col and df_train[c].dtype in [np.float64, np.int64, float, int]]
print(f"  Feature count: {len(feature_cols)}")

def prep(df):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df[label_col].values
    return X, y

X_train_r, y_train_r = prep(df_train)
X_test_r,  y_test_r  = prep(df_test)
X_val_r,   y_val_r   = prep(df_val)

le = LabelEncoder().fit(np.concatenate([y_train_r, y_test_r, y_val_r]))
y_train = le.transform(y_train_r)
y_test  = le.transform(y_test_r)
y_val   = le.transform(y_val_r)
class_names = le.classes_
print(f"  Class count: {len(class_names)}")
print(f"  Classes: {list(class_names)}")

scaler = StandardScaler().fit(X_train_r)
X_train = scaler.transform(X_train_r)
X_test  = scaler.transform(X_test_r)
X_val   = scaler.transform(X_val_r)

# -- 4. TABNET TRAINING -------------------------------------------------
print("\nStarting TabNet training...")
print("  n_steps=5 -> 5 decision steps for 5G slice analysis")

model = TabNetClassifier(
    n_d=32, n_a=32,
    n_steps=5,
    gamma=1.3,
    n_independent=2, n_shared=2,
    momentum=0.02,
    seed=42,
    device_name=DEVICE,
    verbose=1
)

model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_val, y_val)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=50,
    patience=10,
    batch_size=4096,
    virtual_batch_size=256,
    num_workers=0,
    drop_last=False
)

model.save_model("models/tabnet_ciciot2023")
print("OK: models/tabnet_ciciot2023.zip")

# -- 5. PERFORMANCE -----------------------------------------------------
print("\nTest performance...")
y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"  Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
print()
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)  # defined before use
fig, ax = plt.subplots(figsize=(28, 24))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("Actual Label", fontsize=13)
ax.set_title("")
plt.xticks(rotation=90, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("reports/03_confusion_matrix.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()

# -- 6. ATTENTION MASK ANALYSIS -----------------------------------------
print("\nAttention mask analysis (critical for AGE attack)...")

idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
X_sample = X_test[idx]

explain_matrix, masks = model.explain(X_sample)

feat_imp = pd.Series(
    np.mean(np.abs(explain_matrix), axis=0),
    index=feature_cols
).sort_values(ascending=False)

print("\nTop-15 features (primary targets for AGE attack):")
for i, (feat, imp) in enumerate(feat_imp.head(15).items(), 1):
    bar = "X" * int(imp / feat_imp.max() * 20)
    print(f"  {i:2}. {feat:<40} {imp:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(12, 7))
top15 = feat_imp.head(15)
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top15)))
ax.barh(range(len(top15)), top15.values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index[::-1], fontsize=9)
ax.set_xlabel("Mean Attention Weight")
ax.set_title("TabNet Attention Mask -- Top-15 Features\n(Primary Targets for AGE Attack)",
             fontsize=12, fontweight="bold")
ax.axvline(x=top15.values.mean(), color="red", linestyle="--", alpha=0.7, label="Mean")
ax.legend()
plt.tight_layout()
plt.savefig("reports/04_attention_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK: reports/04_attention_feature_importance.png")

print(f"\nDecision step attention ({len(masks)} steps):")
step_imp = []
for si, mask in enumerate(masks):
    mask_arr = mask if isinstance(mask, np.ndarray) else np.array(mask)
    if mask_arr.ndim == 0 or mask_arr.size == 0:
        continue
    # Ensure imp is always 1D with shape (n_features,)
    if mask_arr.ndim == 1:
        imp = np.abs(mask_arr)
    else:
        imp = np.mean(np.abs(mask_arr), axis=0)
    if imp.shape[0] != len(feature_cols):
        continue  # skip malformed masks
    step_imp.append(imp)
    top3 = pd.Series(imp, index=feature_cols).nlargest(3)
    vals = ", ".join([f"{f}({v:.3f})" for f, v in top3.items()])
    print(f"  Step {si+1}: {vals}")

top20_idx = feat_imp.head(20).index.tolist()
top20_pos = [feature_cols.index(f) for f in top20_idx if f in feature_cols]

if len(step_imp) > 0:
    # Stack into 2D array: shape (n_steps, n_features)
    step_matrix = np.vstack(step_imp)[:, top20_pos]

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(step_matrix, cmap="YlOrRd", annot=True, fmt=".3f",
                xticklabels=top20_idx,
                yticklabels=[f"Step {i+1}" for i in range(len(step_imp))],
                ax=ax, linewidths=0.5)
    ax.set_title("TabNet Decision Step x Feature Attention Heatmap\n"
                 "(AGE: Which Step Targets Which Feature?)",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/05_step_attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("OK: reports/05_step_attention_heatmap.png")
else:
    print("WARNING: No valid step attention masks found; skipping heatmap.")

# Save top-5 attention features (required by 04c_age_transfer.py)
feat_imp.head(5).index.to_series().to_csv(
    "reports/top_attention_features.csv", index=False, header=False)
feat_imp.to_csv("reports/full_feature_importance.csv")
print("OK: reports/top_attention_features.csv  (required by AGE attack)")

print("\n" + "="*55)
print("BASELINE TRAINING COMPLETE")
print(f"Baseline accuracy: {acc:.4f}")
print("Next step: python3 04a_fgsm_attack.py")
print("="*55)
