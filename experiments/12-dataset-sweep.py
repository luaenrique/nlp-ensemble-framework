"""
Dataset sweep: runs experiment 11 for every (base, subset) combination,
saving per-window metrics and a summary CSV, plus plots — no plt.show().

Datasets:
  airbnb-comdrift-{1,2,3,4,5}-1-ss.csv
  yelp-comdrift-{1,3,4,5}-1-ss.csv
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — never blocks I/O
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod

from river.drift import ADWIN
from river.drift.binary import DDM

from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ModernBertForSequenceClassification,
)

# ── Output directories ─────────────────────────────────────────────────────────
RESULTS_DIR = "sweep_results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

SUMMARY_CSV  = os.path.join(RESULTS_DIR, "summary.csv")
DETAILED_CSV = os.path.join(RESULTS_DIR, "per_window.csv")

# ── Fixed hyperparams ─────────────────────────────────────────────────────────
DATASET_DIR = "datasets"
DRIFT_TYPE  = 1

BURNIN_SIZE     = 500
N_DRIFTED       = 150_000
WINDOW_SIZE     = 50
N_REF_EMB       = 50
ACC_PLOT_WINDOW = 200   # rolling window size for prequential accuracy display

DRIFT_POSITIONS = [50_000, 100_000, 150_000]

TRAIN_EPOCHS = 3
TRAIN_BATCH  = 16

MMD_THRESHOLD = 0.2
DETECTOR      = "adwin"

ADAPT_SELECT_METHOD = "convex_hull"
ADAPT_KEEP_RATIO    = 0.5
ADAPT_MIN_SAMPLES   = 8

# ── Datasets to sweep ─────────────────────────────────────────────────────────
DATASETS = [
    ("airbnb", 1), ("airbnb", 2), ("airbnb", 3), ("airbnb", 4), ("airbnb", 5),
    ("yelp",   1), ("yelp",   3), ("yelp",   4), ("yelp",   5),
]

# ── Encoders ──────────────────────────────────────────────────────────────────
ENCODERS = [
    {
        "name"              : "ModernBERT",
        "model_name"        : "answerdotai/ModernBERT-base",
        "model_class"       : ModernBertForSequenceClassification,
        "trust_remote_code" : False,
    },
    {
        "name"              : "Jina-v2",
        "model_name"        : "jinaai/jina-embeddings-v2-base-en",
        "model_class"       : AutoModelForSequenceClassification,
        "trust_remote_code" : True,
    },
]

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Device: {device}")

# ── CSV helpers ───────────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    "dataset", "base", "subset", "encoder",
    "n_windows_total", "n_adaptations",
    "first_drift_window_pos",
    "mean_acc", "mean_acc_post_drift",
    "mean_mmd", "mean_kl", "mean_jsd", "mean_centroid",
    "ref_acc", "ref_entropy",
]

DETAIL_FIELDS = [
    "dataset", "base", "subset", "encoder",
    "window_idx", "window_pos",
    "prequential_acc", "entropy", "mmd", "kl", "jsd", "centroid",
    "mmd_warning", "mmd_drift", "jsd_warning", "jsd_drift", "adapted",
]

def _init_csv(path: str, fields: list[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

def _append_row(path: str, fields: list[str], row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)

# ── Metric helpers ─────────────────────────────────────────────────────────────

def _estimate_gamma(embeddings: np.ndarray) -> float:
    if len(embeddings) > 200:
        idx = np.random.choice(len(embeddings), 200, replace=False)
        embeddings = embeddings[idx]
    sq_dists  = pdist(embeddings, metric="sqeuclidean")
    median_sq = float(np.median(sq_dists))
    return 1.0 / (2.0 * median_sq) if median_sq > 0.0 else 1.0


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    X = np.atleast_2d(X).astype(np.float64)
    Y = np.atleast_2d(Y).astype(np.float64)
    n, m = len(X), len(Y)

    def rbf(A, B):
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True)
        return np.exp(-gamma * np.maximum(A_sq + B_sq.T - 2.0 * (A @ B.T), 0.0))

    K_XX, K_YY, K_XY = rbf(X, X), rbf(Y, Y), rbf(X, Y)
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)
    term_XX = K_XX.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_YY = K_YY.sum() / (m * (m - 1)) if m > 1 else 0.0
    return float(max(0.0, term_XX + term_YY - 2.0 * K_XY.mean()))


def compute_centroid_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))


def compute_kl_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-8) -> float:
    P = np.atleast_2d(P).astype(np.float64)
    Q = np.atleast_2d(Q).astype(np.float64)
    mu_p, var_p = P.mean(axis=0), P.var(axis=0) + eps
    mu_q, var_q = Q.mean(axis=0), Q.var(axis=0) + eps
    kl_pq = 0.5 * np.sum(var_p / var_q + (mu_q - mu_p) ** 2 / var_q - 1.0 + np.log(var_q / var_p))
    kl_qp = 0.5 * np.sum(var_q / var_p + (mu_p - mu_q) ** 2 / var_p - 1.0 + np.log(var_p / var_q))
    return float(0.5 * (kl_pq + kl_qp))


def compute_js_divergence(P: np.ndarray, Q: np.ndarray,
                           eps: float = 1e-8, n_samples: int = 2_000) -> float:
    P = np.atleast_2d(P).astype(np.float64)
    Q = np.atleast_2d(Q).astype(np.float64)
    mu_p, var_p = P.mean(axis=0), P.var(axis=0) + eps
    mu_q, var_q = Q.mean(axis=0), Q.var(axis=0) + eps
    std_p, std_q = np.sqrt(var_p), np.sqrt(var_q)
    rng = np.random.default_rng(seed=42)
    z   = rng.standard_normal((n_samples, P.shape[1]))
    S_p = z * std_p + mu_p
    S_q = z * std_q + mu_q

    def lp1d(X, mu, var):
        return -0.5 * (np.log(2.0 * np.pi * var) + (X - mu) ** 2 / var)

    lp_sp = lp1d(S_p, mu_p, var_p); lq_sp = lp1d(S_p, mu_q, var_q)
    lp_sq = lp1d(S_q, mu_p, var_p); lq_sq = lp1d(S_q, mu_q, var_q)
    lm_sp = np.log(0.5) + np.logaddexp(lp_sp, lq_sp)
    lm_sq = np.log(0.5) + np.logaddexp(lp_sq, lq_sq)
    jsd_per_dim = np.clip(0.5 * ((lp_sp - lm_sp).mean(0) + (lq_sq - lm_sq).mean(0)),
                          0.0, np.log(2.0))
    return float(jsd_per_dim.mean())


def _detect_lora_targets(model: torch.nn.Module) -> list[str]:
    candidates = {"query", "key", "value", "q_proj", "k_proj", "v_proj",
                  "Wqkv", "q_lin", "v_lin"}
    found: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in candidates:
                found.add(suffix)
    return list(found) if found else ["query", "value"]


# ── Drift detectors ──────────────────────────────────────────────────────────

class DriftDetectorBase(ABC):
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
    def warning_detected(self) -> bool: return self._warning
    @property
    def drift_detected(self) -> bool:   return self._drift

    def reset(self) -> None:
        self._warn_det  = ADWIN(delta=self._warn_det.delta)
        self._drift_det = ADWIN(delta=self._drift_det.delta)
        self._warning = self._drift = False


class DDMDetector(DriftDetectorBase):
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
    def warning_detected(self) -> bool: return self._warning
    @property
    def drift_detected(self) -> bool:   return self._drift

    def reset(self) -> None:
        self._detector = DDM()
        self._warning = self._drift = False


def make_detector(name: str) -> DriftDetectorBase:
    if name == "adwin": return ADWINDetector()
    if name == "ddm":   return DDMDetector()
    raise ValueError(f"Unknown detector: {name!r}")


# ── Model helpers ─────────────────────────────────────────────────────────────

def encode_and_predict(texts, model, tokenizer, batch_size=32):
    all_emb, all_preds, all_ent = [], [], []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        probas  = F.softmax(outputs.logits, dim=-1)
        entropy = -(probas * torch.log(probas + 1e-10)).sum(dim=-1)
        all_emb.append(outputs.hidden_states[-1][:, 0, :].cpu())
        all_preds.append(outputs.logits.argmax(dim=-1).cpu().numpy())
        all_ent.append(entropy.cpu().numpy())
    return (
        torch.cat(all_emb, dim=0),
        np.concatenate(all_preds),
        np.concatenate(all_ent),
    )


def train_model(model, tokenizer, texts, labels, label_map, label: str = "train"):
    optimizer = AdamW(model.parameters(), lr=2e-4)
    model.train()
    print(f"  [{label.upper()}] {len(texts):,} samples, {TRAIN_EPOCHS} epochs")
    for epoch in range(TRAIN_EPOCHS):
        for i in range(0, len(texts), TRAIN_BATCH):
            bt = texts[i : i + TRAIN_BATCH].tolist()
            bl = torch.tensor([label_map[l] for l in labels[i : i + TRAIN_BATCH]]).to(device)
            inputs = tokenizer(bt, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs, labels=bl).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"  [{label.upper()}] Epoch {epoch+1}/{TRAIN_EPOCHS} done")


def select_drift_samples(ref_np, win_np, X_win, y_win, win_2d,
                          method=ADAPT_SELECT_METHOD,
                          keep_ratio=ADAPT_KEEP_RATIO,
                          min_samples=ADAPT_MIN_SAMPLES):
    n_win  = len(win_np)
    n_keep = max(min_samples, int(keep_ratio * n_win))
    n_keep = min(n_keep, n_win)
    ref_centroid = ref_np.mean(axis=0)

    if method == "convex_hull":
        try:
            hull     = ConvexHull(win_2d)
            hull_idx = set(hull.vertices.tolist())
        except Exception:
            method = "centroid_distance"

    if method == "convex_hull":
        non_hull = [i for i in range(n_win) if i not in hull_idx]
        if len(hull_idx) >= n_keep:
            hull_list  = list(hull_idx)
            hull_dists = np.linalg.norm(win_np[hull_list] - ref_centroid, axis=1)
            selected   = np.array(hull_list)[np.argsort(hull_dists)[::-1][:n_keep]]
        else:
            extra_needed = n_keep - len(hull_idx)
            non_hull_dists = np.linalg.norm(win_np[non_hull] - ref_centroid, axis=1)
            extra    = np.array(non_hull)[np.argsort(non_hull_dists)[-extra_needed:]]
            selected = np.array(list(hull_idx) + extra.tolist())
    else:
        dists    = np.linalg.norm(win_np - ref_centroid, axis=1)
        selected = np.argsort(dists)[-n_keep:]

    selected_indices = np.sort(selected).astype(int)
    return X_win[selected_indices], y_win[selected_indices], selected_indices


# ── Plotting (no show, saves to file) ────────────────────────────────────────

def save_drift_scatter(ref_np, win_np, pos, window_idx, encoder_name, mmd_score,
                        tsne_coords, selected_indices, dataset_tag):
    ref_c = tsne_coords[:len(ref_np)]
    win_c = tsne_coords[len(ref_np):]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ref_c[:, 0], ref_c[:, 1], s=18, alpha=0.6,
               color="steelblue", label=f"Reference (n={len(ref_np)})")

    if selected_indices is not None:
        sel_mask = np.zeros(len(win_np), dtype=bool)
        sel_mask[selected_indices] = True
        ax.scatter(win_c[~sel_mask, 0], win_c[~sel_mask, 1], s=14, alpha=0.25,
                   color="tomato", label=f"Excluded (n={int((~sel_mask).sum())})")
        ax.scatter(win_c[sel_mask, 0], win_c[sel_mask, 1], s=60, alpha=0.9,
                   color="tomato", marker="*", label=f"Hull selected (n={int(sel_mask.sum())})")
    else:
        ax.scatter(win_c[:, 0], win_c[:, 1], s=18, alpha=0.6,
                   color="tomato", label=f"Drift (n={len(win_np)})")

    ax.set_title(f"Drift — {dataset_tag} / {encoder_name}\npos {pos:,}  MMD={mmd_score:.4f}", fontsize=11)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fname = os.path.join(PLOTS_DIR, f"scatter_{dataset_tag}_{encoder_name.lower()}_w{window_idx}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SCATTER] → {fname}")


def save_results_plot(
    window_positions, window_accuracies, window_entropies,
    mmd_scores, kl_scores, js_scores, centroid_scores,
    adapt_positions, jsd_warn_positions, jsd_drift_positions,
    ref_acc, ref_entropy,
    encoder_name, dataset_tag, base, subset,
):
    drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]
    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(
        f"Concept Drift — {base}-comdrift-{subset}-{DRIFT_TYPE}-ss.csv\n"
        f"{encoder_name} + LoRA  |  Window = {WINDOW_SIZE:,} samples",
        fontsize=13,
    )

    def _fill(ax, ys, color, label, ref=None, ref_label=None, ylabel=""):
        ax.plot(window_positions, ys, color=color, linewidth=1.3, label=label)
        ax.fill_between(window_positions, ys, alpha=0.15, color=color)
        if ref is not None:
            ax.axhline(ref, color=color, linestyle=":", linewidth=1, alpha=0.7, label=ref_label)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

    _fill(axes[0], window_accuracies, "steelblue",
          f"Prequential acc (rolling {ACC_PLOT_WINDOW})",
          ref=ref_acc, ref_label=f"Ref acc ({ref_acc:.3f})", ylabel="Accuracy")
    axes[0].set_ylim(0, 1.05)

    _fill(axes[1], window_entropies, "seagreen", "Mean entropy",
          ref=ref_entropy, ref_label=f"Ref entropy ({ref_entropy:.4f})", ylabel="Entropy (nats)")

    _fill(axes[2], mmd_scores, "darkorange", "MMD",
          ref=MMD_THRESHOLD, ref_label=f"Threshold ({MMD_THRESHOLD})", ylabel="MMD")

    _fill(axes[3], kl_scores,  "crimson",  "Sym KL div", ylabel="KL div")
    _fill(axes[4], js_scores,  "teal",     "JSD",
          ref=np.log(2), ref_label=f"Max JSD ({np.log(2):.3f})", ylabel="JSD (nats)")
    _fill(axes[5], centroid_scores, "mediumpurple", "Centroid dist", ylabel="Centroid dist")

    axes[5].set_xlabel("Position in comdrift stream", fontsize=11)

    for pos in adapt_positions:
        for ax in axes:
            ax.axvline(x=pos, color="limegreen", linestyle=":", linewidth=1.2, alpha=0.75)
    for pos in jsd_warn_positions:
        axes[4].axvline(x=pos, color="gold",    linestyle="--", linewidth=1.2, alpha=0.8)
    for pos in jsd_drift_positions:
        axes[4].axvline(x=pos, color="darkcyan", linestyle="--", linewidth=1.5, alpha=0.9)

    legend_patches = []
    for pos, col in zip(DRIFT_POSITIONS, drift_colors):
        for ax in axes:
            ax.axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
        legend_patches.append(mpatches.Patch(color=col, label=f"Drift @ {pos//1_000}k"))
    if adapt_positions:
        legend_patches.append(mpatches.Patch(color="limegreen", label="LoRA adapted"))

    x_ticks = np.arange(0, N_DRIFTED + 1, 25_000)
    axes[5].set_xticks(x_ticks)
    axes[5].set_xticklabels([f"{x//1_000}k" for x in x_ticks])
    axes[5].set_xlim(0, N_DRIFTED)
    axes[5].legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"metrics_{dataset_tag}_{encoder_name.lower()}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] → {fname}")


# ── Main sweep ────────────────────────────────────────────────────────────────

_init_csv(SUMMARY_CSV,  SUMMARY_FIELDS)
_init_csv(DETAILED_CSV, DETAIL_FIELDS)

for base, subset in DATASETS:
    com_path = f"{DATASET_DIR}/{base}-comdrift-{subset}-{DRIFT_TYPE}-ss.csv"
    dataset_tag = f"{base}-{subset}-{DRIFT_TYPE}"
    print(f"\n{'#'*70}")
    print(f"  Dataset: {com_path}")
    print(f"{'#'*70}")

    com_df = pd.read_csv(com_path).dropna(subset=["review_treated"])
    all_texts  = com_df["review_treated"].astype(str).values[:N_DRIFTED]
    all_labels = com_df["label"].values[:N_DRIFTED]

    burnin_texts  = all_texts[:BURNIN_SIZE]
    burnin_labels = all_labels[:BURNIN_SIZE]
    stream_texts  = all_texts[BURNIN_SIZE:]
    stream_labels = all_labels[BURNIN_SIZE:]
    n_windows     = len(stream_texts) // WINDOW_SIZE

    label_map     = {v: i for i, v in enumerate(np.unique(burnin_labels))}
    inv_label_map = {i: v for v, i in label_map.items()}
    num_labels    = len(label_map)

    print(f"  Burn-in: {len(burnin_texts):,}  |  Stream: {len(stream_texts):,}  |  Windows: {n_windows:,}")

    for enc in ENCODERS:
        print(f"\n{'='*60}")
        print(f"  Encoder: {enc['name']}  |  Dataset: {dataset_tag}")
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
        lora_config  = LoraConfig(r=8, lora_alpha=16, target_modules=lora_targets,
                                  task_type=TaskType.SEQ_CLS)
        model = get_peft_model(model, lora_config).to(device)
        model.print_trainable_parameters()

        train_model(model, tokenizer, burnin_texts, burnin_labels, label_map, label="train")

        ref_embs_t, ref_preds, ref_entropies_arr = encode_and_predict(
            burnin_texts[:N_REF_EMB], model, tokenizer
        )
        ref_embs_t = ref_embs_t.to(device)
        ref_np     = ref_embs_t.cpu().numpy()

        ref_preds_orig = np.array([inv_label_map[p] for p in ref_preds])
        ref_acc        = (ref_preds_orig == burnin_labels[:N_REF_EMB]).mean()
        ref_entropy    = ref_entropies_arr.mean()
        print(f"  Reference acc: {ref_acc:.3f}  entropy: {ref_entropy:.4f}")

        window_positions  = []
        window_accuracies = []
        window_entropies  = []
        mmd_scores        = []
        kl_scores         = []
        js_scores         = []
        centroid_scores   = []
        adapt_positions   = []
        jsd_warn_positions  = []
        jsd_drift_positions = []

        gamma        = _estimate_gamma(ref_np)
        detector     = make_detector(DETECTOR)
        jsd_detector = make_detector(DETECTOR)
        first_drift_pos = None
        correct_history = []   # prequential: 1/0 per item across entire stream

        for i in range(n_windows):
            start = i * WINDOW_SIZE
            end   = start + WINDOW_SIZE
            X_win = stream_texts[start:end]
            y_win = stream_labels[start:end]

            win_embs_t, win_preds, win_ent = encode_and_predict(X_win, model, tokenizer)
            win_np = win_embs_t.cpu().numpy()

            preds_orig = np.array([inv_label_map[p] for p in win_preds])

            # prequential: record each item's correctness in stream order
            correct_history.extend((preds_orig == y_win).astype(int).tolist())
            # rolling accuracy over the last ACC_PLOT_WINDOW items
            acc    = float(np.mean(correct_history[-ACC_PLOT_WINDOW:]))
            entropy = win_ent.mean()
            score         = compute_mmd(ref_np, win_np, gamma)
            kl            = compute_kl_divergence(ref_np, win_np)
            js            = compute_js_divergence(ref_np, win_np)
            centroid_dist = compute_centroid_distance(ref_np, win_np)
            pos           = BURNIN_SIZE + start + WINDOW_SIZE // 2

            detector.update(score)
            jsd_detector.update(js)

            mmd_warn  = detector.warning_detected
            mmd_drift = detector.drift_detected
            jsd_warn  = jsd_detector.warning_detected
            jsd_drift = jsd_detector.drift_detected
            adapted   = False

            if jsd_warn:
                jsd_warn_positions.append(pos)
            if jsd_drift:
                jsd_drift_positions.append(pos)

            if mmd_drift:
                if first_drift_pos is None:
                    first_drift_pos = pos
                print(f"  [DRIFT] window {i+1} pos {pos:,}  mmd={score:.4f}")

                from sklearn.manifold import TSNE
                combined    = np.vstack([ref_np, win_np])
                tsne_coords = TSNE(n_components=2,
                                   perplexity=min(30, len(combined) // 4),
                                   random_state=42, n_jobs=-1).fit_transform(combined)
                win_2d = tsne_coords[len(ref_np):]

                X_sel, y_sel, sel_idx = select_drift_samples(
                    ref_np, win_np, X_win, y_win, win_2d,
                    method=ADAPT_SELECT_METHOD,
                    keep_ratio=ADAPT_KEEP_RATIO,
                    min_samples=ADAPT_MIN_SAMPLES,
                )
                save_drift_scatter(ref_np, win_np, pos, i + 1, enc["name"], score,
                                   tsne_coords, sel_idx, dataset_tag)

                train_model(model, tokenizer, X_sel, y_sel, label_map, label="adapt")

                new_ref_t, new_ref_preds, new_ref_ent = encode_and_predict(X_win, model, tokenizer)
                ref_np      = new_ref_t.cpu().numpy()
                ref_entropy = new_ref_ent.mean()
                new_preds_orig = np.array([inv_label_map[p] for p in new_ref_preds])
                ref_acc     = (new_preds_orig == y_win).mean()
                gamma       = _estimate_gamma(ref_np)
                detector.reset()
                jsd_detector.reset()
                adapt_positions.append(pos)
                adapted = True

            window_positions.append(pos)
            window_accuracies.append(acc)
            window_entropies.append(entropy)
            mmd_scores.append(score)
            kl_scores.append(kl)
            js_scores.append(js)
            centroid_scores.append(centroid_dist)

            _append_row(DETAILED_CSV, DETAIL_FIELDS, {
                "dataset": dataset_tag, "base": base, "subset": subset,
                "encoder": enc["name"], "window_idx": i + 1, "window_pos": pos,
                "prequential_acc": round(acc, 4), "entropy": round(float(entropy), 4),
                "mmd": round(score, 6), "kl": round(kl, 6),
                "jsd": round(js, 6), "centroid": round(centroid_dist, 6),
                "mmd_warning": int(mmd_warn), "mmd_drift": int(mmd_drift),
                "jsd_warning": int(jsd_warn), "jsd_drift": int(jsd_drift),
                "adapted": int(adapted),
            })

            if (i + 1) % 100 == 0:
                print(f"  [{enc['name']}] {i+1}/{n_windows}  acc={acc:.3f}  mmd={score:.4f}")

        save_results_plot(
            window_positions, window_accuracies, window_entropies,
            mmd_scores, kl_scores, js_scores, centroid_scores,
            adapt_positions, jsd_warn_positions, jsd_drift_positions,
            ref_acc, ref_entropy,
            enc["name"], dataset_tag, base, subset,
        )

        post_drift_accs = [
            window_accuracies[j]
            for j, p in enumerate(window_positions)
            if p >= DRIFT_POSITIONS[0]
        ]

        _append_row(SUMMARY_CSV, SUMMARY_FIELDS, {
            "dataset":               dataset_tag,
            "base":                  base,
            "subset":                subset,
            "encoder":               enc["name"],
            "n_windows_total":       n_windows,
            "n_adaptations":         len(adapt_positions),
            "first_drift_window_pos": first_drift_pos if first_drift_pos else "none",
            "mean_acc":              round(float(np.mean(window_accuracies)), 4),
            "mean_acc_post_drift":   round(float(np.mean(post_drift_accs)), 4) if post_drift_accs else "n/a",
            "mean_mmd":              round(float(np.mean(mmd_scores)), 6),
            "mean_kl":               round(float(np.mean(kl_scores)), 6),
            "mean_jsd":              round(float(np.mean(js_scores)), 6),
            "mean_centroid":         round(float(np.mean(centroid_scores)), 6),
            "ref_acc":               round(float(ref_acc), 4),
            "ref_entropy":           round(float(ref_entropy), 4),
        })

        print(f"  [DONE] {dataset_tag} / {enc['name']}  adaptations={len(adapt_positions)}")

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

print(f"\nSweep complete.")
print(f"  Summary  → {SUMMARY_CSV}")
print(f"  Details  → {DETAILED_CSV}")
print(f"  Plots    → {PLOTS_DIR}/")
