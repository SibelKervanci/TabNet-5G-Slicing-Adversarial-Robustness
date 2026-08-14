"""
fig_confusion_matrix.py
Kaydedilmiş modeli yükler, confusion matrix çizer.
Çalıştırma: python fig_confusion_matrix.py
Çıktı: reports/03_confusion_matrix.png
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("OK: TabNet hazır")
except ImportError:
    print("HATA: pip install pytorch-tabnet"); exit(1)

# ── MODEL YÜKLE ──────────────────────────────────────────────────────────────
model = TabNetClassifier()
model.load_model("models/tabnet_ciciot2023.zip")
print("OK: Model yüklendi")

# ── VERİ YÜKLE ───────────────────────────────────────────────────────────────
print("Test verisi yükleniyor...")
df = pd.read_csv(os.path.join("CICIOT2023", "test", "test.csv"),
                 low_memory=False, nrows=50000)

label_col    = "label"
feature_cols = [c for c in df.columns
                if c != label_col and df[c].dtype in [np.float64, np.int64, float, int]]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
y_str = df[label_col].values

le = LabelEncoder()
y  = le.fit_transform(y_str)
class_names = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── TAHMİN ───────────────────────────────────────────────────────────────────
print("Tahmin yapılıyor...")
y_pred = model.predict(X_scaled)
acc    = (y_pred == y).mean()
print(f"Accuracy: {acc:.4f}")

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots(figsize=(36, 30))
fig.patch.set_facecolor("white")

sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax, linewidths=0.3)

ax.set_xlabel("Predicted Label", fontsize=16)
ax.set_ylabel("True Label", fontsize=16)
ax.set_title("")

plt.xticks(rotation=90, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

plt.savefig("reports/03_confusion_matrix.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("OK: reports/03_confusion_matrix.png")
