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
from scipy.spatial.distance import pdist

from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertForSequenceClassification

DATASET_DIR = "datasets"
BASE        = "airbnb"
SUBSET      = 1
DRIFT_TYPE  = 2

N_REFERENCE  = 50_000     # rows from semdrift used to train the classifier
N_DRIFTED    = 150_000    # rows from comdrift to stream through
WINDOW_SIZE  = 400      # evaluation window (samples per accuracy point)

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

n_windows        = len(stream_texts) // WINDOW_SIZE
label_map        = {v: i for i, v in enumerate(np.unique(ref_labels))}
inv_label_map    = {i: v for v, i in label_map.items()}

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ── Model + LoRA ──────────────────────────────────────────────────────────────
num_labels = len(label_map)

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = ModernBertForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base", num_labels=num_labels
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["Wqkv"],
    task_type=TaskType.SEQ_CLS,
)

model = get_peft_model(model, lora_config).to(device)
model.print_trainable_parameters()

# ── Train classifier on reference window ─────────────────────────────────────
TRAIN_EPOCHS = 3
TRAIN_BATCH  = 16

optimizer = AdamW(model.parameters(), lr=2e-4)
model.train()

for epoch in range(TRAIN_EPOCHS):
    for i in range(0, len(ref_texts), TRAIN_BATCH):
        batch_texts  = ref_texts[i : i + TRAIN_BATCH].tolist()
        batch_labels = torch.tensor(
            [label_map[l] for l in ref_labels[i : i + TRAIN_BATCH]]
        ).to(device)

        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        loss = model(**inputs, labels=batch_labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS} done")

print("Training complete.")


def l2_norm(t):
    return t / t.norm(dim=1, keepdim=True)


# def MMD(x, y, kernel):
#     """Emprical maximum mean discrepancy. The lower the result
#        the more evidence that distributions are the same.
#
#     Args:
#         x: first sample, distribution P
#         y: second sample, distribution Q
#         kernel: kernel type such as "multiscale" or "rbf"
#     """
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))
#
#     dxx = rx.t() + rx - 2. * xx # Used for A in (1)
#     dyy = ry.t() + ry - 2. * yy # Used for B in (1)
#     dxy = rx.t() + ry - 2. * zz # Used for C in (1)
#
#     XX, YY, XY = (torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device))
#
#     if kernel == "multiscale":
#         bandwidth_range = [0.2, 0.5, 0.9, 1.3]
#         for a in bandwidth_range:
#             XX += a**2 * (a**2 + dxx)**-1
#             YY += a**2 * (a**2 + dyy)**-1
#             XY += a**2 * (a**2 + dxy)**-1
#
#     if kernel == "rbf":
#         bandwidth_range = [10, 15, 20, 50]
#         for a in bandwidth_range:
#             XX += torch.exp(-0.5*dxx/a)
#             YY += torch.exp(-0.5*dyy/a)
#             XY += torch.exp(-0.5*dxy/a)
#
#     return torch.mean(XX + YY - 2. * XY)


def _estimate_gamma(embeddings: np.ndarray) -> float:
    """
    Median heuristic: gamma = 1 / (2 * median_pairwise_distance²).

    Estimating gamma from data is important for high-dimensional embeddings
    (e.g. 768-d BERT/Jina) where the default gamma=1.0 would produce kernel
    values near zero, making MMD blind to any distributional shift.
    """
    if len(embeddings) > 200:
        idx = np.random.choice(len(embeddings), 200, replace=False)
        embeddings = embeddings[idx]

    sq_dists = pdist(embeddings, metric="sqeuclidean")
    median_sq = float(np.median(sq_dists))
    return 1.0 / (2.0 * median_sq) if median_sq > 0.0 else 1.0


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """
    Unbiased estimate of MMD² between sample sets X and Y using the RBF kernel.

        MMD²(P, Q) = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
        k(a, b)    = exp(-gamma * ||a - b||²)

    Args:
        X     : Reference embeddings  shape (n, d)
        Y     : Detection embeddings  shape (m, d)
        gamma : RBF bandwidth (estimated from data via _estimate_gamma)

    Returns:
        MMD² score (float >= 0)
    """
    X = np.atleast_2d(X).astype(np.float64)
    Y = np.atleast_2d(Y).astype(np.float64)
    n, m = len(X), len(Y)

    def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True)
        dist_sq = A_sq + B_sq.T - 2.0 * (A @ B.T)
        return np.exp(-gamma * np.maximum(dist_sq, 0.0))

    K_XX = rbf(X, X)
    K_YY = rbf(Y, Y)
    K_XY = rbf(X, Y)

    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)

    term_XX = K_XX.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_YY = K_YY.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_XY = K_XY.mean()

    return float(max(0.0, term_XX + term_YY - 2.0 * term_XY))


def compute_centroid_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """L2 distance between the mean embedding of X and Y."""
    return float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))


def encode_and_predict(texts, batch_size=32):
    all_embeddings, all_preds, all_entropies = [], [], []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        probas   = F.softmax(outputs.logits, dim=-1)
        entropy  = -(probas * torch.log(probas + 1e-10)).sum(dim=-1)

        all_embeddings.append(outputs.hidden_states[-1][:, 0, :].cpu())
        all_preds.append(outputs.logits.argmax(dim=-1).cpu().numpy())
        all_entropies.append(entropy.cpu().numpy())

    embeddings = torch.cat(all_embeddings, dim=0)
    preds      = np.concatenate(all_preds, axis=0)
    entropies  = np.concatenate(all_entropies, axis=0)
    return embeddings, preds, entropies



# ── Reference embeddings + accuracy baseline ─────────────────────────────────
N_REF_EMB = 400
ref_embs_t, ref_preds, ref_entropies = encode_and_predict(ref_texts[:N_REF_EMB])
ref_embs_t = ref_embs_t.to(device)

ref_preds_orig = np.array([inv_label_map[p] for p in ref_preds])
ref_acc        = (ref_preds_orig == ref_labels[:N_REF_EMB]).mean()
ref_entropy    = ref_entropies.mean()
print(f"Reference accuracy (ModernBERT): {ref_acc:.3f}  |  entropy: {ref_entropy:.4f}")

sigma = torch.cdist(ref_embs_t, ref_embs_t).median().item()
print(f"Median pairwise distance (ref): {sigma:.4f}")

# ── Stream evaluation ─────────────────────────────────────────────────────────
window_positions  = []
window_accuracies = []
window_entropies  = []
mmd_scores        = []
centroid_scores   = []

for i in range(n_windows):
    start_batch_pos = i * WINDOW_SIZE
    end_batch_pos   = start_batch_pos + WINDOW_SIZE

    X_win = stream_texts[start_batch_pos:end_batch_pos]
    y_win = stream_labels[start_batch_pos:end_batch_pos]

    win_embs_t, win_preds, win_entropies = encode_and_predict(X_win)
    win_embs_t = win_embs_t.to(device)

    preds_orig = np.array([inv_label_map[p] for p in win_preds])
    acc     = (preds_orig == y_win).mean()
    entropy = win_entropies.mean()
    ref_np        = ref_embs_t.cpu().numpy()
    win_np        = win_embs_t.cpu().numpy()
    gamma         = _estimate_gamma(ref_np)
    score         = compute_mmd(ref_np, win_np, gamma)
    centroid_dist = compute_centroid_distance(ref_np, win_np)

    window_accuracies.append(acc)
    window_entropies.append(entropy)
    mmd_scores.append(score)
    centroid_scores.append(centroid_dist)

    window_positions.append(start_batch_pos + WINDOW_SIZE // 2)
    print(f"Window {i+1}/{n_windows}  acc={acc:.3f}  entropy={entropy:.4f}  mmd={score:.4f}  centroid={centroid_dist:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]

fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.suptitle(
    f"Concept Drift — {BASE}-comdrift-{SUBSET}-{DRIFT_TYPE}.csv\n"
    f"ModernBERT + LoRA  |  Window = {WINDOW_SIZE:,} samples",
    fontsize=13,
)

ax1 = axes[0]
ax1.plot(window_positions, window_accuracies, color="steelblue", linewidth=1.3, label="Window accuracy")
ax1.fill_between(window_positions, window_accuracies, alpha=0.15, color="steelblue")
ax1.axhline(ref_acc, color="steelblue", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Reference accuracy ({ref_acc:.3f})")
ax1.set_ylabel("Accuracy", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower left", fontsize=9)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(window_positions, window_entropies, color="seagreen", linewidth=1.3, label="Mean prediction entropy")
ax2.fill_between(window_positions, window_entropies, alpha=0.15, color="seagreen")
ax2.axhline(ref_entropy, color="seagreen", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Reference entropy ({ref_entropy:.4f})")
ax2.set_ylabel("Entropy (nats)", fontsize=11)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.3)

ax3 = axes[2]
ax3.plot(window_positions, mmd_scores, color="darkorange", linewidth=1.3, label="MMD score")
ax3.fill_between(window_positions, mmd_scores, alpha=0.15, color="darkorange")
ax3.set_ylabel("MMD score", fontsize=11)
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(alpha=0.3)

ax4 = axes[3]
ax4.plot(window_positions, centroid_scores, color="mediumpurple", linewidth=1.3, label="Centroid distance")
ax4.fill_between(window_positions, centroid_scores, alpha=0.15, color="mediumpurple")
ax4.set_ylabel("Centroid distance", fontsize=11)
ax4.set_xlabel("Position in comdrift stream (sample index)", fontsize=11)
ax4.legend(loc="upper left", fontsize=9)
ax4.grid(alpha=0.3)

legend_patches = []
for pos, col in zip(DRIFT_POSITIONS, drift_colors):
    label = f"Drift @ {pos // 1_000}k"
    for ax in axes:
        ax.axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
    legend_patches.append(mpatches.Patch(color=col, label=label))

x_ticks = np.arange(0, N_DRIFTED + 1, 25_000)
ax4.set_xticks(x_ticks)
ax4.set_xticklabels([f"{x // 1_000}k" for x in x_ticks])
axes[3].legend(handles=legend_patches, loc="upper right", fontsize=9)

plt.tight_layout()
out_path = "mmd_drift_detection_4.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved → {out_path}")