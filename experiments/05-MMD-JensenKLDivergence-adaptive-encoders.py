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
  - Burn-in window   : first BURNIN_SIZE samples from comdrift (assumed stable)
                       → used for LoRA fine-tuning and as reference distribution
  - Detection stream : remaining comdrift samples (comdrift[BURNIN_SIZE:N_DRIFTED])
  - Window positions : absolute indices in the comdrift file
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

BURNIN_SIZE  = 500   # first N samples from comdrift used for training + reference
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
        "out_path"          : "exp_05_drift_modernbert_ss.png",
    },
    {
        "name"              : "Jina-v2",
        "model_name"        : "jinaai/jina-embeddings-v2-base-en",
        "model_class"       : AutoModelForSequenceClassification,
        "trust_remote_code" : True,
        "out_path"          : "exp_05drift_jina_ss.png",
    },
]

# ── Load data ──────────────────────────────────────────────────────────────────
com_path = f"{DATASET_DIR}/{BASE}-comdrift-{SUBSET}-{DRIFT_TYPE}-ss.csv"

com_df = pd.read_csv(com_path).dropna(subset=["review_treated"])

all_texts  = com_df["review_treated"].astype(str).values[:N_DRIFTED]
all_labels = com_df["label"].values[:N_DRIFTED]

# Burn-in window: first BURNIN_SIZE samples (assumed pre-drift, stable)
burnin_texts  = all_texts[:BURNIN_SIZE]
burnin_labels = all_labels[:BURNIN_SIZE]

# Detection stream: everything after the burn-in window
stream_texts  = all_texts[BURNIN_SIZE:]
stream_labels = all_labels[BURNIN_SIZE:]

print(f"Burn-in   : {len(burnin_texts):>7,} samples  |  labels: {np.unique(burnin_labels)}")
print(f"Stream    : {len(stream_texts):>7,} samples  |  labels: {np.unique(stream_labels)}")

n_windows     = len(stream_texts) // WINDOW_SIZE
label_map     = {v: i for i, v in enumerate(np.unique(burnin_labels))}
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


def compute_kl_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric KL divergence between two empirical distributions under a
    diagonal Gaussian approximation.

    Each distribution is represented by its per-dimension mean and variance
    (i.e. we assume feature independence). This avoids inverting a
    768×768 covariance matrix and is numerically stable.

        KL(P||Q) = 0.5 * Σ_d [ σ_P²/σ_Q² + (μ_Q-μ_P)²/σ_Q² - 1 + ln(σ_Q²/σ_P²) ]
        KL_sym   = 0.5 * (KL(P||Q) + KL(Q||P))

    Args:
        P   : Reference embeddings  shape (n, d)
        Q   : Window embeddings     shape (m, d)
        eps : Small constant for numerical stability

    Returns:
        Symmetric KL divergence (float >= 0)
    """
    P = np.atleast_2d(P).astype(np.float64)
    Q = np.atleast_2d(Q).astype(np.float64)

    mu_p, var_p = P.mean(axis=0), P.var(axis=0) + eps
    mu_q, var_q = Q.mean(axis=0), Q.var(axis=0) + eps

    kl_pq = 0.5 * np.sum(var_p / var_q + (mu_q - mu_p) ** 2 / var_q - 1.0 + np.log(var_q / var_p))
    kl_qp = 0.5 * np.sum(var_q / var_p + (mu_p - mu_q) ** 2 / var_p - 1.0 + np.log(var_p / var_q))

    return float(0.5 * (kl_pq + kl_qp))


def compute_js_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-8, n_samples: int = 2_000) -> float:
    """
    Jensen-Shannon divergence estimated via Monte Carlo under a diagonal Gaussian
    approximation, consistent with compute_kl_divergence.

        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)   where M = 0.5*(P+Q)

    Since the mixture M is not Gaussian, KL(·||M) has no closed form.
    We sample from the fitted Gaussians and estimate via log-likelihood ratios:

        KL(P||M) ≈ E_{x~P}[ log p(x) - log m(x) ]
        log m(x) = log(0.5) + logaddexp(log p(x), log q(x))

    Result is in [0, ln(2)] ≈ [0, 0.693].

    Args:
        P        : Reference embeddings  shape (n, d)
        Q        : Window embeddings     shape (m, d)
        eps      : Variance floor for numerical stability
        n_samples: MC sample size per distribution

    Returns:
        JSD estimate (float in [0, ln(2)])
    """
    P = np.atleast_2d(P).astype(np.float64)
    Q = np.atleast_2d(Q).astype(np.float64)

    mu_p, var_p = P.mean(axis=0), P.var(axis=0) + eps
    mu_q, var_q = Q.mean(axis=0), Q.var(axis=0) + eps
    std_p, std_q = np.sqrt(var_p), np.sqrt(var_q)

    rng = np.random.default_rng(seed=42)
    S_p = rng.normal(mu_p, std_p, (n_samples, P.shape[1]))
    S_q = rng.normal(mu_q, std_q, (n_samples, Q.shape[1]))

    def log_prob(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        return -0.5 * np.sum(np.log(2.0 * np.pi * var) + (X - mu) ** 2 / var, axis=1)

    log_p_sp = log_prob(S_p, mu_p, var_p)
    log_q_sp = log_prob(S_p, mu_q, var_q)
    log_p_sq = log_prob(S_q, mu_p, var_p)
    log_q_sq = log_prob(S_q, mu_q, var_q)

    log_m_sp = np.log(0.5) + np.logaddexp(log_p_sp, log_q_sp)
    log_m_sq = np.log(0.5) + np.logaddexp(log_p_sq, log_q_sq)

    kl_p_m = float((log_p_sp - log_m_sp).mean())
    kl_q_m = float((log_q_sq - log_m_sq).mean())

    return float(np.clip(0.5 * (kl_p_m + kl_q_m), 0.0, np.log(2.0)))


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
    all_embeddings, all_preds, all_entropies = [], [], []
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

        all_embeddings.append(outputs.hidden_states[-1][:, 0, :].cpu())
        all_preds.append(outputs.logits.argmax(dim=-1).cpu().numpy())
        all_entropies.append(entropy.cpu().numpy())

    return (
        torch.cat(all_embeddings, dim=0),
        np.concatenate(all_preds),
        np.concatenate(all_entropies),
    )


def train_model(model, tokenizer, texts, labels):
    optimizer = AdamW(model.parameters(), lr=2e-4)
    model.train()

    for epoch in range(TRAIN_EPOCHS):
        for i in range(0, len(texts), TRAIN_BATCH):
            batch_texts  = texts[i : i + TRAIN_BATCH].tolist()
            batch_labels = torch.tensor(
                [label_map[l] for l in labels[i : i + TRAIN_BATCH]]
            ).to(device)

            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            loss = model(**inputs, labels=batch_labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"  Epoch {epoch + 1}/{TRAIN_EPOCHS} done")


def plot_results(
    window_positions, window_accuracies, window_entropies,
    mmd_scores, kl_scores, js_scores, centroid_scores,
    ref_acc, ref_entropy,
    encoder_name, out_path,
):
    drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(
        f"Concept Drift — {BASE}-comdrift-{SUBSET}-{DRIFT_TYPE}.csv\n"
        f"{encoder_name} + LoRA  |  Window = {WINDOW_SIZE:,} samples",
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
    ax4.plot(window_positions, kl_scores, color="crimson", linewidth=1.3, label="Symmetric KL divergence")
    ax4.fill_between(window_positions, kl_scores, alpha=0.15, color="crimson")
    ax4.set_ylabel("KL divergence", fontsize=11)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(alpha=0.3)

    ax5 = axes[4]
    ax5.plot(window_positions, js_scores, color="teal", linewidth=1.3, label="Jensen-Shannon divergence")
    ax5.fill_between(window_positions, js_scores, alpha=0.15, color="teal")
    ax5.set_ylabel("JSD (nats)", fontsize=11)
    ax5.legend(loc="upper left", fontsize=9)
    ax5.grid(alpha=0.3)

    ax6 = axes[5]
    ax6.plot(window_positions, centroid_scores, color="mediumpurple", linewidth=1.3, label="Centroid distance")
    ax6.fill_between(window_positions, centroid_scores, alpha=0.15, color="mediumpurple")
    ax6.set_ylabel("Centroid distance", fontsize=11)
    ax6.set_xlabel("Position in comdrift stream (sample index)", fontsize=11)
    ax6.legend(loc="upper left", fontsize=9)
    ax6.grid(alpha=0.3)

    legend_patches = []
    for pos, col in zip(DRIFT_POSITIONS, drift_colors):
        label = f"Drift @ {pos // 1_000}k"
        for ax in axes:
            ax.axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
        legend_patches.append(mpatches.Patch(color=col, label=label))

    x_ticks = np.arange(0, N_DRIFTED + 1, 25_000)
    ax6.set_xticks(x_ticks)
    ax6.set_xticklabels([f"{x // 1_000}k" for x in x_ticks])
    ax6.set_xlim(0, N_DRIFTED)
    axes[5].legend(handles=legend_patches, loc="upper right", fontsize=9)

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

    print("Training on burn-in window …")
    train_model(model, tokenizer, burnin_texts, burnin_labels)
    print("Training complete.")

    # ── Reference embeddings + accuracy baseline (from burn-in window) ─────────
    ref_embs_t, ref_preds, ref_entropies = encode_and_predict(
        burnin_texts[:N_REF_EMB], model, tokenizer
    )
    ref_embs_t = ref_embs_t.to(device)

    ref_preds_orig = np.array([inv_label_map[p] for p in ref_preds])
    ref_acc        = (ref_preds_orig == burnin_labels[:N_REF_EMB]).mean()
    ref_entropy    = ref_entropies.mean()
    print(f"Reference accuracy: {ref_acc:.3f}  |  entropy: {ref_entropy:.4f}")

    ref_np = ref_embs_t.cpu().numpy()

    # ── Stream evaluation ──────────────────────────────────────────────────────
    window_positions  = []
    window_accuracies = []
    window_entropies  = []
    mmd_scores        = []
    kl_scores         = []
    js_scores         = []
    centroid_scores   = []

    gamma = _estimate_gamma(ref_np)  # estimate once from reference

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end   = start + WINDOW_SIZE

        X_win = stream_texts[start:end]
        y_win = stream_labels[start:end]

        win_embs_t, win_preds, win_entropies = encode_and_predict(X_win, model, tokenizer)
        win_embs_t = win_embs_t.to(device)
        win_np     = win_embs_t.cpu().numpy()

        preds_orig    = np.array([inv_label_map[p] for p in win_preds])
        acc           = (preds_orig == y_win).mean()
        entropy       = win_entropies.mean()
        score         = compute_mmd(ref_np, win_np, gamma)
        kl            = compute_kl_divergence(ref_np, win_np)
        js            = compute_js_divergence(ref_np, win_np)
        centroid_dist = compute_centroid_distance(ref_np, win_np)

        window_accuracies.append(acc)
        window_entropies.append(entropy)
        mmd_scores.append(score)
        kl_scores.append(kl)
        js_scores.append(js)
        centroid_scores.append(centroid_dist)
        window_positions.append(BURNIN_SIZE + start + WINDOW_SIZE // 2)

        print(f"  [{enc['name']}] Window {i+1}/{n_windows}  acc={acc:.3f}  entropy={entropy:.4f}  mmd={score:.4f}  kl={kl:.4f}  jsd={js:.4f}  centroid={centroid_dist:.4f}")

    plot_results(
        window_positions, window_accuracies, window_entropies,
        mmd_scores, kl_scores, js_scores, centroid_scores,
        ref_acc, ref_entropy,
        enc["name"], enc["out_path"],
    )

    # Free memory before loading the next encoder
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
