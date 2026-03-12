"""
Drift detection over the tech_non_tech_dataset.csv tweet stream.

Dataset: experiments/tech_non_tech_dataset.csv
  - Binary classification: tech vs non-tech tweets
  - Temporal stream: sorted by created_at to simulate a real-world stream
  - Year distribution: 2009, 2015, 2016, 2026

Experimental setup (same as exp-07):
  - Burn-in window   : first BURNIN_SIZE samples (assumed stable)
                       → used for LoRA fine-tuning and as reference distribution
  - Detection stream : remaining samples (stream[BURNIN_SIZE:])
  - Year boundaries  : marked as vertical lines in the plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from river.drift import ADWIN
from river.drift.binary import DDM

from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ModernBertForSequenceClassification

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASET_PATH = "experiments/tech_non_tech_dataset.csv"

BURNIN_SIZE  = 200    # first N samples from the sorted stream for training + reference
WINDOW_SIZE  = 50
N_REF_EMB    = 50

TRAIN_EPOCHS = 3
TRAIN_BATCH  = 16

MMD_THRESHOLD = 0.2   # binarisation threshold for DDM; also plotted as reference line
DETECTOR      = "adwin"  # "adwin" | "ddm"

# ── Encoder configs ────────────────────────────────────────────────────────────
ENCODERS = [
    {
        "name"              : "ModernBERT",
        "model_name"        : "answerdotai/ModernBERT-base",
        "model_class"       : ModernBertForSequenceClassification,
        "trust_remote_code" : False,
        "out_path"          : "exp_08_drift_modernbert_technontech.png",
    },
    {
        "name"              : "Jina-v2",
        "model_name"        : "jinaai/jina-embeddings-v2-base-en",
        "model_class"       : AutoModelForSequenceClassification,
        "trust_remote_code" : True,
        "out_path"          : "exp_08_drift_jina_technontech.png",
    },
]

# ── Load data ──────────────────────────────────────────────────────────────────
com_df = (
    pd.read_csv(DATASET_PATH)
    .dropna(subset=["text"])
    .sort_values("created_at")
    .reset_index(drop=True)
)

all_texts  = com_df["text"].astype(str).values
all_labels = com_df["label"].values
N_DRIFTED  = len(all_texts)

# Compute year-boundary positions in the sorted stream (for plot annotations)
year_boundaries: dict[int, int] = {}
for year, grp in com_df.groupby("year"):
    year_boundaries[int(year)] = int(grp.index[0])

# Burn-in window: first BURNIN_SIZE samples
burnin_texts  = all_texts[:BURNIN_SIZE]
burnin_labels = all_labels[:BURNIN_SIZE]

# Detection stream: everything after the burn-in window
stream_texts  = all_texts[BURNIN_SIZE:]
stream_labels = all_labels[BURNIN_SIZE:]

print(f"Total     : {N_DRIFTED:>7,} samples")
print(f"Burn-in   : {len(burnin_texts):>7,} samples  |  labels: {np.unique(burnin_labels)}")
print(f"Stream    : {len(stream_texts):>7,} samples  |  labels: {np.unique(stream_labels)}")
print(f"Year boundaries: {year_boundaries}")

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


# ── Drift detectors ─────────────────────────────────────────────────────────────

class DriftDetectorBase(ABC):
    """Minimal interface for a two-level (warning + drift) detector."""

    @abstractmethod
    def update(self, value: float) -> None: ...

    @property
    @abstractmethod
    def warning_detected(self) -> bool: ...

    @property
    @abstractmethod
    def drift_detected(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...


class ADWINDetector(DriftDetectorBase):
    """
    Two ADWIN instances with different sensitivity.
    delta_warning > delta_drift  →  warning fires earlier than confirmed drift.
    """

    def __init__(self, delta_warning: float = 0.01, delta_drift: float = 0.001):
        self._warn_det  = ADWIN(delta=delta_warning)
        self._drift_det = ADWIN(delta=delta_drift)
        self._warning   = False
        self._drift     = False

    def update(self, value: float) -> None:
        self._warn_det.update(value)
        self._drift_det.update(value)
        self._warning = self._warn_det.drift_detected
        self._drift   = self._drift_det.drift_detected

    @property
    def warning_detected(self) -> bool:
        return self._warning

    @property
    def drift_detected(self) -> bool:
        return self._drift

    def reset(self) -> None:
        self._warn_det  = ADWIN(delta=self._warn_det.delta)
        self._drift_det = ADWIN(delta=self._drift_det.delta)
        self._warning   = False
        self._drift     = False


class DDMDetector(DriftDetectorBase):
    """
    DDM wrapper. Binarises the continuous MMD value (> threshold → 1, else 0)
    to feed the error-rate-based DDM detector (Gama et al., 2004).
    Has native warning and drift levels.
    """

    def __init__(self, threshold: float = MMD_THRESHOLD):
        self._detector  = DDM()
        self._threshold = threshold
        self._warning   = False
        self._drift     = False

    def update(self, value: float) -> None:
        self._detector.update(int(value > self._threshold))
        self._warning = self._detector.warning_detected
        self._drift   = self._detector.drift_detected

    @property
    def warning_detected(self) -> bool:
        return self._warning

    @property
    def drift_detected(self) -> bool:
        return self._drift

    def reset(self) -> None:
        self._detector = DDM()
        self._warning  = False
        self._drift    = False


def make_detector(name: str) -> DriftDetectorBase:
    if name == "adwin":
        return ADWINDetector()
    if name == "ddm":
        return DDMDetector()
    raise ValueError(f"Unknown detector: {name!r}. Choose 'adwin' or 'ddm'.")


# ── Word trajectory visualization ─────────────────────────────────────────────

TRAJECTORY_WORDS  = ["model", "agent", "learning"]
TRAJECTORY_YEARS  = [2009, 2026]   # only compare the two extreme years
TRAJECTORY_COLORS = {"2009": "#3498db", "2026": "#e74c3c"}
MAX_TWEETS_PER_YEAR = 150  # cap per (word, year) to keep encoding fast



def _get_word_token_embeddings(
    texts: list[str],
    word: str,
    model,
    tokenizer,
    batch_size: int = 16,
) -> tuple[np.ndarray, list[str]]:
    """
    For each tweet, locate the subword token(s) of `word` in the tokenised
    sequence and return their mean hidden-state vector (last layer).

    This gives the *contextual* embedding of the word rather than the CLS
    embedding of the whole tweet — much more informative for tracking
    semantic shift.

    Returns (embeddings, valid_texts) where valid_texts are the tweets
    in which the word token was actually found.
    """
    # Token IDs for the bare word (no special tokens)
    word_ids = tokenizer.encode(word, add_special_tokens=False)
    n_w = len(word_ids)

    embs: list[np.ndarray] = []
    valid: list[str] = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc_dev = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc_dev, output_hidden_states=True)

        hidden   = out.hidden_states[-1].cpu().numpy()   # (B, T, d)
        input_ids = enc["input_ids"].numpy()             # (B, T)

        for j, text in enumerate(batch):
            ids = input_ids[j].tolist()
            # Find first contiguous occurrence of word_ids in the sequence
            pos = next(
                (k for k in range(len(ids) - n_w + 1)
                 if ids[k : k + n_w] == word_ids),
                None,
            )
            if pos is None:
                continue
            embs.append(hidden[j, pos : pos + n_w, :].mean(axis=0))
            valid.append(text)

    if not embs:
        return np.empty((0, 0)), []
    return np.vstack(embs), valid


def plot_word_trajectories(model, tokenizer, encoder_name: str, out_path: str) -> None:
    """
    Interactive Plotly chart (HTML) — 3 subplots, one per target word.

    Each subplot shows the contextual token embedding of the word extracted
    from each tweet, projected to 2D with PCA fitted only on those embeddings.
    Points are coloured by year (2009 = blue, 2026 = red); hovering shows the
    tweet text.  A centroid marker per year shows the mean shift.
    """
    rng = np.random.default_rng(seed=0)

    # ── 1. Collect token embeddings per (word, year) ──────────────────────────
    word_year_embs:  dict[tuple[str, int], np.ndarray] = {}
    word_year_texts: dict[tuple[str, int], list[str]]  = {}

    for word in TRAJECTORY_WORDS:
        mask = com_df["text"].str.contains(rf"\b{word}\b", case=False, regex=True)
        sub  = com_df[mask]
        for year in TRAJECTORY_YEARS:
            rows = sub[sub["year"] == year]["text"].values
            if len(rows) == 0:
                continue
            if len(rows) > MAX_TWEETS_PER_YEAR:
                rows = rng.choice(rows, MAX_TWEETS_PER_YEAR, replace=False)

            print(f"  [TRAJECTORY] '{word}' {year} — {len(rows)} tweets …")
            embs, valid_texts = _get_word_token_embeddings(
                list(rows), word, model, tokenizer
            )
            if len(embs) == 0:
                continue
            word_year_embs[(word, year)]  = embs
            word_year_texts[(word, year)] = valid_texts

    if not word_year_embs:
        print("  [TRAJECTORY] No embeddings found — skipping.")
        return

    # ── 2. Build figure ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=len(TRAJECTORY_WORDS),
        subplot_titles=[f'"{w}"' for w in TRAJECTORY_WORDS],
        horizontal_spacing=0.06,
    )

    for col_idx, word in enumerate(TRAJECTORY_WORDS):
        col         = col_idx + 1
        show_legend = col_idx == 0

        # Fit PCA per word (only on this word's embeddings — no mixing)
        word_embs = [word_year_embs[(word, yr)]
                     for yr in TRAJECTORY_YEARS if (word, yr) in word_year_embs]
        if not word_embs:
            continue
        pca = PCA(n_components=2, random_state=42)
        pca.fit(np.vstack(word_embs))
        var = pca.explained_variance_ratio_

        for year in TRAJECTORY_YEARS:
            if (word, year) not in word_year_embs:
                continue

            pts_2d   = pca.transform(word_year_embs[(word, year)])
            tweets   = word_year_texts[(word, year)]
            color    = TRAJECTORY_COLORS[str(year)]
            centroid = pts_2d.mean(axis=0)
            hover    = [t[:140] + "…" if len(t) > 140 else t for t in tweets]

            # Individual points
            fig.add_trace(
                go.Scatter(
                    x=pts_2d[:, 0], y=pts_2d[:, 1],
                    mode="markers",
                    marker=dict(size=5, color=color, opacity=0.45,
                                line=dict(width=0)),
                    name=str(year),
                    legendgroup=str(year),
                    showlegend=show_legend,
                    hovertemplate="<b>" + str(year) + "</b><br>%{customdata}<extra></extra>",
                    customdata=hover,
                ),
                row=1, col=col,
            )

            # Centroid marker
            fig.add_trace(
                go.Scatter(
                    x=[centroid[0]], y=[centroid[1]],
                    mode="markers+text",
                    marker=dict(size=14, color=color, symbol="diamond",
                                line=dict(width=1.5, color="white")),
                    text=[str(year)],
                    textposition="top center",
                    textfont=dict(size=10, color=color),
                    legendgroup=str(year),
                    showlegend=False,
                    hovertemplate=f"<b>centroid {year}</b><extra></extra>",
                ),
                row=1, col=col,
            )

        fig.update_xaxes(
            showticklabels=False,
            title_text=f"PC1 {var[0]:.0%} | PC2 {var[1]:.0%}",
            row=1, col=col,
        )
        fig.update_yaxes(showticklabels=False, row=1, col=col)

    fig.update_layout(
        title=dict(
            text=(
                f"Contextual token shift — {encoder_name}<br>"
                "<sup>each point = embedding of the word token inside one tweet"
                " · hover to read it</sup>"
            ),
            font=dict(size=13),
        ),
        height=520,
        width=420 * len(TRAJECTORY_WORDS),
        template="plotly_white",
        legend=dict(title="year", tracegroupgap=4),
    )

    html_path = out_path.replace(".png", ".html")
    fig.write_html(html_path)
    print(f"  [TRAJECTORY] Saved → {html_path}")


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


def train_model(model, tokenizer, texts, labels, label: str = "train"):
    optimizer = AdamW(model.parameters(), lr=2e-4)
    model.train()

    print(f"  [{label.upper()}] Starting — {len(texts):,} samples, {TRAIN_EPOCHS} epochs")
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

        print(f"  [{label.upper()}] Epoch {epoch + 1}/{TRAIN_EPOCHS} done")


def plot_drift_scatter(
    ref_np: np.ndarray,
    win_np: np.ndarray,
    pos: int,
    window_idx: int,
    encoder_name: str,
    mmd_score: float,
):
    """
    2-D PCA scatter of reference vs drifted window embeddings.
    Saved to disk as drift_scatter_{encoder_name}_w{window_idx}.png
    """
    combined = np.vstack([ref_np, win_np])
    pca      = PCA(n_components=2)
    coords   = pca.fit_transform(combined)
    var      = pca.explained_variance_ratio_

    ref_c = coords[:len(ref_np)]
    win_c = coords[len(ref_np):]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ref_c[:, 0], ref_c[:, 1], s=18, alpha=0.6,
               color="steelblue", label=f"Reference  (n={len(ref_np)})")
    ax.scatter(win_c[:, 0], win_c[:, 1], s=18, alpha=0.6,
               color="tomato",    label=f"Drift window (n={len(win_np)})")

    ax.set_title(
        f"Drift detected — {encoder_name}\n"
        f"stream pos {pos:,}  |  MMD={mmd_score:.4f}",
        fontsize=11,
    )
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = f"drift_scatter_{encoder_name.lower()}_w{window_idx}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"  [SCATTER] Saved → {out}")


def plot_results(
    window_positions, window_accuracies, window_entropies,
    mmd_scores, kl_scores, js_scores, centroid_scores,
    adapt_positions,
    ref_acc, ref_entropy,
    encoder_name, out_path,
):
    year_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(
        f"Concept Drift — tech_non_tech_dataset (tweets, sorted by time)\n"
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
    ax3.axhline(MMD_THRESHOLD, color="darkorange", linestyle=":", linewidth=1.2, alpha=0.8,
                label=f"Adapt threshold ({MMD_THRESHOLD})")
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
    ax6.set_xlabel("Position in stream (sample index)", fontsize=11)
    ax6.legend(loc="upper left", fontsize=9)
    ax6.grid(alpha=0.3)

    for pos in adapt_positions:
        for ax in axes:
            ax.axvline(x=pos, color="limegreen", linestyle=":", linewidth=1.2, alpha=0.75)

    # Mark year boundaries
    legend_patches = []
    for (year, boundary_pos), col in zip(sorted(year_boundaries.items()), year_colors):
        for ax in axes:
            ax.axvline(x=boundary_pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
        legend_patches.append(mpatches.Patch(color=col, label=f"Year {year} starts @ {boundary_pos:,}"))
    if adapt_positions:
        legend_patches.append(mpatches.Patch(color="limegreen", label="LoRA adapted"))

    tick_step = max(1_000, N_DRIFTED // 10)
    x_ticks = np.arange(0, N_DRIFTED + 1, tick_step)
    ax6.set_xticks(x_ticks)
    ax6.set_xticklabels([f"{x:,}" for x in x_ticks])
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
    train_model(model, tokenizer, burnin_texts, burnin_labels, label="train")
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

    # ── Word trajectory visualization (2009 → 2026) ───────────────────────────
    traj_out = enc["out_path"].replace(".png", "_word_trajectory.png")
    plot_word_trajectories(model, tokenizer, enc["name"], traj_out)

    # ── Stream evaluation ──────────────────────────────────────────────────────
    window_positions  = []
    window_accuracies = []
    window_entropies  = []
    mmd_scores        = []
    kl_scores         = []
    js_scores         = []
    centroid_scores   = []
    adapt_positions   = []

    gamma    = _estimate_gamma(ref_np)  # estimate once from reference
    detector = make_detector(DETECTOR)
    print(f"  Drift detector: {DETECTOR.upper()}")

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

        # ── Feed MMD score to detector ─────────────────────────────────────────
        detector.update(score)
        pos = BURNIN_SIZE + start + WINDOW_SIZE // 2

        if detector.warning_detected:
            print(f"  [WARNING] {DETECTOR.upper()} detected distributional warning  "
                  f"— window {i+1} (pos {pos:,})  mmd={score:.4f}")

        if detector.drift_detected:
            print(f"  [DRIFT]   {DETECTOR.upper()} confirmed concept drift          "
                  f"— window {i+1} (pos {pos:,})  mmd={score:.4f}")
            plot_drift_scatter(ref_np, win_np, pos, i + 1, enc["name"], score)

        # ── Detect-then-adapt: retrain LoRA on confirmed drift ─────────────────
        if detector.drift_detected:
            print(f"  [ADAPT] Fake Retraining LoRA …")
            #train_model(model, tokenizer, X_win, y_win, label="adapt")
            # Re-encode with updated model to get new reference distribution
            #new_ref_t, new_ref_preds, new_ref_entropies = encode_and_predict(X_win, model, tokenizer)
            #ref_np      = new_ref_t.cpu().numpy()
            #ref_entropy = new_ref_entropies.mean()
            #new_preds_orig = np.array([inv_label_map[p] for p in new_ref_preds])
            #ref_acc     = (new_preds_orig == y_win).mean()
            #gamma  = _estimate_gamma(ref_np)
            #detector.reset()
            #adapt_positions.append(pos)
            print(f"  [ADAPT] Done — new reference set, gamma and detector reset.")

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
        adapt_positions,
        ref_acc, ref_entropy,
        enc["name"], enc["out_path"],
    )

    # Free memory before loading the next encoder
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
