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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ModernBertForSequenceClassification

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASET_DIR = "datasets"
BASE        = "airbnb"
SUBSET      = 2
DRIFT_TYPE  = 1

N_REFERENCE  = 50_000
N_DRIFTED    = 150_000
WINDOW_SIZE  = 400
N_REF_EMB    = 400

DRIFT_POSITIONS = [50_000, 100_000, 150_000]

TRAIN_EPOCHS = 3
TRAIN_BATCH  = 16

# ── Encoder configs ────────────────────────────────────────────────────────────
ENCODERS = [
    {
        "name"              : "ModernBERT",
        "model_name"        : "answerdotai/ModernBERT-base",
        "model_class"       : ModernBertForSequenceClassification,
        "trust_remote_code" : False,
        "out_path"          : "drift_modernbert_ss_exp_3.png",
    },
    {
        "name"              : "Jina-v2",
        "model_name"        : "jinaai/jina-embeddings-v2-base-en",
        "model_class"       : AutoModelForSequenceClassification,
        "trust_remote_code" : True,
        "out_path"          : "drift_jina_ss_exp_3.png",
    },
]

# ── Load data ──────────────────────────────────────────────────────────────────
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

n_windows     = len(stream_texts) // WINDOW_SIZE
label_map     = {v: i for i, v in enumerate(np.unique(ref_labels))}
inv_label_map = {i: v for v, i in label_map.items()}
num_labels    = len(label_map)

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ── Metric helpers ─────────────────────────────────────────────────────────────

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


def _detect_lora_targets(model: torch.nn.Module) -> list[str]:
    """
    Auto-detect suitable LoRA target module names by scanning the model's
    Linear layers for common attention-projection suffixes.
    """
    candidates = {
        "query", "key", "value",
        "q_proj", "k_proj", "v_proj",
        "Wqkv", "q_lin", "v_lin",
    }
    found: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in candidates:
                found.add(suffix)
    return list(found) if found else ["query", "value"]


# ── Old MMD (biased, torch, multiscale kernel) — kept for reference ────────────
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

# ── Encoder helpers ────────────────────────────────────────────────────────────

def encode_and_predict(texts, model, tokenizer, batch_size=32):
    """
    Returns:
        all_layer_embs : np.ndarray  (n_layers, n_samples, hidden_dim)
        preds          : np.ndarray  (n_samples,)
        entropies      : np.ndarray  (n_samples,)

    hidden_states[0] is the input embedding layer — skipped.
    hidden_states[1:] are the transformer layers (indexed from 1).
    """
    all_layer_embs = None  # will be (n_layers, n_samples, hidden_dim)
    all_preds      = []
    all_entropies  = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        probas  = F.softmax(outputs.logits, dim=-1)
        entropy = -(probas * torch.log(probas + 1e-10)).sum(dim=-1)

        # Stack CLS token from each transformer layer: (n_layers, batch, hidden_dim)
        batch_layers = torch.stack(
            [h[:, 0, :].cpu() for h in outputs.hidden_states[1:]], dim=0
        )

        all_layer_embs = batch_layers if all_layer_embs is None \
                         else torch.cat([all_layer_embs, batch_layers], dim=1)
        all_preds.append(outputs.logits.argmax(dim=-1).cpu().numpy())
        all_entropies.append(entropy.cpu().numpy())

    return (
        all_layer_embs.numpy(),        # (n_layers, n_samples, hidden_dim)
        np.concatenate(all_preds),     # (n_samples,)
        np.concatenate(all_entropies), # (n_samples,)
    )


def train_model(model, tokenizer):
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

        print(f"  Epoch {epoch + 1}/{TRAIN_EPOCHS} done")


def _normalize_cols(M: np.ndarray) -> np.ndarray:
    """Normalize each column (layer) of M to [0, 1] across windows."""
    mn = M.min(axis=0, keepdims=True)
    mx = M.max(axis=0, keepdims=True)
    return (M - mn) / np.where(mx - mn > 0, mx - mn, 1.0)


def plot_results(
    window_positions, window_accuracies, window_entropies,
    mmd_matrix, centroid_matrix,
    ref_acc, ref_entropy,
    encoder_name, out_path,
):
    """
    4-panel figure:
      Panel 1 (line)    : accuracy over time
      Panel 2 (line)    : entropy over time
      Panel 3 (heatmap) : MMD per layer × window position
      Panel 4 (heatmap) : centroid distance per layer × window position
    """
    drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]
    n_layers     = mmd_matrix.shape[1]
    x_min, x_max = window_positions[0], window_positions[-1]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"Concept Drift — {BASE}-comdrift-{SUBSET}-{DRIFT_TYPE}.csv\n"
        f"{encoder_name} + LoRA  |  Window = {WINDOW_SIZE:,} samples",
        fontsize=13,
    )

    # ── Panel 1: Accuracy ──────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(window_positions, window_accuracies, color="steelblue", linewidth=1.3, label="Window accuracy")
    ax1.fill_between(window_positions, window_accuracies, alpha=0.15, color="steelblue")
    ax1.axhline(ref_acc, color="steelblue", linestyle=":", linewidth=1, alpha=0.7,
                label=f"Reference accuracy ({ref_acc:.3f})")
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Entropy ───────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(window_positions, window_entropies, color="seagreen", linewidth=1.3, label="Mean prediction entropy")
    ax2.fill_between(window_positions, window_entropies, alpha=0.15, color="seagreen")
    ax2.axhline(ref_entropy, color="seagreen", linestyle=":", linewidth=1, alpha=0.7,
                label=f"Reference entropy ({ref_entropy:.4f})")
    ax2.set_ylabel("Entropy (nats)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)

    # ── Panel 3: MMD heatmap ───────────────────────────────────────────────────
    ax3 = axes[2]
    norm_mmd = _normalize_cols(mmd_matrix)   # (n_windows, n_layers)
    im3 = ax3.imshow(
        norm_mmd.T,                           # (n_layers, n_windows)
        aspect="auto", origin="lower",
        cmap="YlOrRd",
        extent=[x_min, x_max, 0.5, n_layers + 0.5],
    )
    ax3.set_ylabel("Layer", fontsize=11)
    ax3.set_yticks(range(1, n_layers + 1, max(1, n_layers // 8)))
    plt.colorbar(im3, ax=ax3, label="MMD (normalized per layer)", pad=0.01)

    # ── Panel 4: Centroid heatmap ──────────────────────────────────────────────
    ax4 = axes[3]
    norm_centroid = _normalize_cols(centroid_matrix)
    im4 = ax4.imshow(
        norm_centroid.T,
        aspect="auto", origin="lower",
        cmap="PuRd",
        extent=[x_min, x_max, 0.5, n_layers + 0.5],
    )
    ax4.set_ylabel("Layer", fontsize=11)
    ax4.set_yticks(range(1, n_layers + 1, max(1, n_layers // 8)))
    ax4.set_xlabel("Position in comdrift stream (sample index)", fontsize=11)
    plt.colorbar(im4, ax=ax4, label="Centroid distance (normalized per layer)", pad=0.01)

    # ── Drift markers ──────────────────────────────────────────────────────────
    legend_patches = []
    for pos, col in zip(DRIFT_POSITIONS, drift_colors):
        axes[0].axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
        axes[1].axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
        axes[2].axvline(x=pos, color="white", linestyle="--", linewidth=1.5, alpha=0.9)
        axes[3].axvline(x=pos, color="white", linestyle="--", linewidth=1.5, alpha=0.9)
        legend_patches.append(mpatches.Patch(color=col, label=f"Drift @ {pos // 1_000}k"))

    x_ticks = np.arange(0, N_DRIFTED + 1, 25_000)
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels([f"{x // 1_000}k" for x in x_ticks])

    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved → {out_path}")


# ── Main loop over encoders ────────────────────────────────────────────────────
for enc in ENCODERS:
    print(f"\n{'='*60}")
    print(f"  Encoder: {enc['name']}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        enc["model_name"], trust_remote_code=enc["trust_remote_code"]
    )
    model = enc["model_class"].from_pretrained(
        enc["model_name"],
        num_labels=num_labels,
        trust_remote_code=enc["trust_remote_code"],
    )

    lora_targets = _detect_lora_targets(model)
    print(f"  LoRA targets detected: {lora_targets}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=lora_targets,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config).to(device)
    model.print_trainable_parameters()

    print("Training …")
    train_model(model, tokenizer)
    print("Training complete.")

    # ── Reference embeddings + accuracy baseline ───────────────────────────────
    ref_embs_all, ref_preds, ref_entropies = encode_and_predict(
        ref_texts[:N_REF_EMB], model, tokenizer
    )
    # ref_embs_all: (n_layers, N_REF_EMB, hidden_dim)

    n_layers = ref_embs_all.shape[0]
    print(f"  Hidden layers detected: {n_layers}  |  shape: {ref_embs_all.shape}")

    ref_preds_orig = np.array([inv_label_map[p] for p in ref_preds])
    ref_acc        = (ref_preds_orig == ref_labels[:N_REF_EMB]).mean()
    ref_entropy    = ref_entropies.mean()
    print(f"  Reference accuracy: {ref_acc:.3f}  |  entropy: {ref_entropy:.4f}")

    # Estimate gamma once per layer from the reference embeddings
    ref_gammas = [_estimate_gamma(ref_embs_all[l]) for l in range(n_layers)]

    # ── Stream evaluation ──────────────────────────────────────────────────────
    window_positions  = []
    window_accuracies = []
    window_entropies  = []
    mmd_matrix        = []  # (n_windows, n_layers)
    centroid_matrix   = []  # (n_windows, n_layers)

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end   = start + WINDOW_SIZE

        X_win = stream_texts[start:end]
        y_win = stream_labels[start:end]

        win_embs_all, win_preds, win_entropies = encode_and_predict(X_win, model, tokenizer)
        # win_embs_all: (n_layers, WINDOW_SIZE, hidden_dim)

        preds_orig = np.array([inv_label_map[p] for p in win_preds])
        acc        = (preds_orig == y_win).mean()
        entropy    = win_entropies.mean()

        mmd_row      = []
        centroid_row = []
        for l in range(n_layers):
            mmd_row.append(compute_mmd(ref_embs_all[l], win_embs_all[l], ref_gammas[l]))
            centroid_row.append(compute_centroid_distance(ref_embs_all[l], win_embs_all[l]))

        window_accuracies.append(acc)
        window_entropies.append(entropy)
        mmd_matrix.append(mmd_row)
        centroid_matrix.append(centroid_row)
        window_positions.append(start + WINDOW_SIZE // 2)

        print(f"  [{enc['name']}] Window {i+1}/{n_windows}  acc={acc:.3f}  entropy={entropy:.4f}  mmd_last={mmd_row[-1]:.4f}")

    mmd_matrix      = np.array(mmd_matrix)      # (n_windows, n_layers)
    centroid_matrix = np.array(centroid_matrix)  # (n_windows, n_layers)
    print(f"  mmd_matrix shape: {mmd_matrix.shape}")

    plot_results(
        window_positions, window_accuracies, window_entropies,
        mmd_matrix, centroid_matrix,
        ref_acc, ref_entropy,
        enc["name"], enc["out_path"],
    )

    # Free memory before loading the next encoder
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
