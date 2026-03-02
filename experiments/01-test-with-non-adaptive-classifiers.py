"""
Drift detection over text streams from structured CSV datasets.

Dataset naming convention:
  {base}-[comdrift|semdrift]-{subset}-{drift_type}.csv

  base       : "airbnb" or "yelp"
  comdrift   : data WITH concept drift
  semdrift   : data WITHOUT drift (reference distribution)
  subset     : 1–5 (variability between runs)
  drift_type : 1, 1-ss, 2, 3 (type of drift applied)

Experimental setup:
  - Reference window : texts from semdrift file (stable distribution)
  - Detection stream : texts from comdrift file (drifted distribution)
  - Full stream      : semdrift_texts[:N_REFERENCE] + comdrift_texts[:N_DRIFTED]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

DATASET_DIR = "../datasets"
BASE        = "airbnb"
SUBSET      = 1
DRIFT_TYPE  = 2

N_REFERENCE  = 50_000     # rows from semdrift used to train the classifier
N_DRIFTED    = 150_000    # rows from comdrift to stream through
WINDOW_SIZE  = 2_000      # evaluation window (samples per accuracy point)

DRIFT_POSITIONS = [50_000, 100_000, 150_000]

# ── Load data ─────────────────────────────────────────────────────────────────
sem_path = f"{DATASET_DIR}/{BASE}-semdrift-{SUBSET}-1.csv"
com_path = f"{DATASET_DIR}/{BASE}-comdrift-{SUBSET}-{DRIFT_TYPE}.csv"

sem_df = pd.read_csv(sem_path).dropna(subset=["review_treated"])
com_df = pd.read_csv(com_path).dropna(subset=["review_treated"])

ref_texts  = sem_df["review_treated"].astype(str).values[:N_REFERENCE]
ref_labels = sem_df["label"].values[:N_REFERENCE]

stream_texts  = com_df["review_treated"].astype(str).values[:N_DRIFTED]
stream_labels = com_df["label"].values[:N_DRIFTED]

print(f"Reference : {len(ref_texts):>7,} samples  |  labels: {np.unique(ref_labels)}")
print(f"Stream    : {len(stream_texts):>7,} samples  |  labels: {np.unique(stream_labels)}")

clf = Pipeline([
    ("vec", HashingVectorizer(
        n_features=2 ** 16,
        ngram_range=(1, 2),
        alternate_sign=False,
    )),
    ("cls", SGDClassifier(
        loss="log_loss",
        max_iter=5,
        random_state=42,
        n_jobs=-1,
    )),
])

print("\nTraining on reference window …")
clf.fit(ref_texts, ref_labels)
ref_acc = (clf.predict(ref_texts) == ref_labels).mean()
print(f"Reference train accuracy: {ref_acc:.3f}")

n_windows         = len(stream_texts) // WINDOW_SIZE
window_positions  = [] 
window_accuracies = []
window_entropies  = []  

classes = np.unique(ref_labels)

for i in range(n_windows):
    start = i * WINDOW_SIZE
    end   = start + WINDOW_SIZE

    X_win = stream_texts[start:end]
    y_win = stream_labels[start:end]

    probas = clf.predict_proba(X_win)          # (window, n_classes)
    preds  = classes[probas.argmax(axis=1)]

    acc = (preds == y_win).mean()

    # Entropy H = -sum(p * log(p+eps)) per sample, then averaged
    eps     = 1e-10
    entropy = -(probas * np.log(probas + eps)).sum(axis=1).mean()

    window_positions.append(start + WINDOW_SIZE // 2)
    window_accuracies.append(acc)
    window_entropies.append(entropy)

window_positions  = np.array(window_positions)
window_accuracies = np.array(window_accuracies)
window_entropies  = np.array(window_entropies)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(
    f"Concept Drift Impact — {BASE}-comdrift-{SUBSET}-{DRIFT_TYPE}.csv\n"
    f"Classifier: SGD (log-loss) + HashingVectorizer  |  Window = {WINDOW_SIZE:,} samples",
    fontsize=13,
)

drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]

# Panel 1 – Accuracy
ax1 = axes[0]
ax1.plot(window_positions, window_accuracies,
         color="steelblue", linewidth=1.3, label="Window accuracy")
ax1.fill_between(window_positions, window_accuracies, alpha=0.15, color="steelblue")
ax1.axhline(ref_acc, color="steelblue", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Reference accuracy ({ref_acc:.3f})")
ax1.set_ylabel("Accuracy", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower left", fontsize=9)
ax1.grid(alpha=0.3)

# Panel 2 – Entropy
ax2 = axes[1]
ax2.plot(window_positions, window_entropies,
         color="darkorange", linewidth=1.3, label="Mean prediction entropy")
ax2.fill_between(window_positions, window_entropies, alpha=0.15, color="darkorange")
ax2.set_ylabel("Entropy (nats)", fontsize=11)
ax2.set_xlabel("Position in comdrift stream (sample index)", fontsize=11)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.3)

# Drift markers
legend_patches = []
for pos, col in zip(DRIFT_POSITIONS, drift_colors):
    label = f"Drift @ {pos // 1_000}k"
    for ax in axes:
        ax.axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
    legend_patches.append(mpatches.Patch(color=col, label=label))

# x-axis tick labels in thousands
x_ticks = np.arange(0, N_DRIFTED + 1, 25_000)
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([f"{x // 1_000}k" for x in x_ticks])

axes[1].legend(handles=legend_patches, loc="upper right", fontsize=9)

plt.tight_layout()
out_path = "drift_classification_impact.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved → {out_path}")
