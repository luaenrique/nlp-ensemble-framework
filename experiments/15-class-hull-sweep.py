"""
Experiment 15 — Convex Hull per Class + Full Grid Sweep

Compares two sample-selection strategies at drift time:
  convex_hull           : convex hull over the full window (t-SNE 2D)
  convex_hull_per_class : convex hull applied separately per class (t-SNE 2D),
                          then concatenated → LoRA fine-tune

Grid:
  datasets          : DATASETS list
  encoders          : ModernBERT, Jina-v2
  selection_methods : convex_hull, convex_hull_per_class
  window_sizes      : [50, 100, 200]
  detectors         : DDM, EDDM, ADWIN
  metrics           : MMD, JSD

Total runs: 5 × 2 × 2 × 3 × 3 × 2 = 360

Detection logic:
  - warning → start filling warning buffer with incoming windows
  - drift   → select samples (via chosen method), retrain LoRA,
               reset buffer and detector

Outputs (per run):
  experiment15_results/per_window/{tag}.csv
  experiment15_results/plots/{tag}.png
  experiment15_results/summary.csv
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import csv
import copy
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from abc import ABC, abstractmethod
from sklearn.manifold import TSNE

from river.drift import ADWIN
from river.drift.binary import DDM, EDDM

from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ModernBertForSequenceClassification,
)

warnings.filterwarnings("ignore")

# ── Output dirs ────────────────────────────────────────────────────────────────
RESULTS_DIR    = "experiment15_results"
PLOTS_DIR      = os.path.join(RESULTS_DIR, "plots")
PER_WINDOW_DIR = os.path.join(RESULTS_DIR, "per_window")
SUMMARY_CSV    = os.path.join(RESULTS_DIR, "summary.csv")

for d in [RESULTS_DIR, PLOTS_DIR, PER_WINDOW_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Fixed hyperparams ──────────────────────────────────────────────────────────
DATASET_DIR = "datasets"
DRIFT_TYPE  = 1

BURNIN_SIZE     = 500
N_DRIFTED       = 150_000
DRIFT_POSITIONS = [50_000, 100_000, 150_000]

ACC_PLOT_WINDOW_FACTOR = 4

BURNIN_EPOCHS = 3
ADAPT_EPOCHS  = 1
TRAIN_BATCH   = 16

MMD_THRESHOLD = 0.20
JSD_THRESHOLD = 0.10

ADAPT_KEEP_RATIO   = 0.5
ADAPT_MIN_SAMPLES  = 8
WARNING_BUFFER_MAX = 500

# ── Sweep grid ─────────────────────────────────────────────────────────────────
SWEEP_WINDOW_SIZES      = [50, 100, 200]
SWEEP_DETECTORS         = ["adwin", "ddm", "eddm"]
SWEEP_METRICS           = ["mmd", "jsd"]
SWEEP_SELECTION_METHODS = ["convex_hull", "convex_hull_per_class"]

DATASETS = [
    ("airbnb", 1), ("airbnb", 2), ("airbnb", 3), ("airbnb", 4), ("airbnb", 5),
]

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

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

# ── CSV helpers ────────────────────────────────────────────────────────────────
SUMMARY_FIELDS = [
    "run_tag",
    "dataset", "base", "subset", "encoder",
    "selection_method", "detector", "metric", "window_size",
    "wall_time_s",
    "n_windows", "n_warnings", "n_drifts", "n_adaptations",
    "first_drift_pos",
    "mean_acc", "mean_acc_post_drift",
    "mean_mmd", "mean_jsd", "mean_kl", "mean_centroid",
    "ref_acc_initial", "ref_entropy_initial",
]

DETAIL_FIELDS = [
    "window_idx", "window_pos",
    "prequential_acc", "entropy",
    "mmd", "kl", "jsd", "centroid",
    "warning", "drift", "adapted",
    "warning_buffer_size",
]


def _init_summary_csv() -> None:
    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()


def _append_summary(row: dict) -> None:
    with open(SUMMARY_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow(row)


def _save_per_window_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        w.writeheader()
        w.writerows(rows)


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
    t_XX = K_XX.sum() / (n * (n - 1)) if n > 1 else 0.0
    t_YY = K_YY.sum() / (m * (m - 1)) if m > 1 else 0.0
    return float(max(0.0, t_XX + t_YY - 2.0 * K_XY.mean()))


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
                           eps: float = 1e-8, n_samples: int = 500) -> float:
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
    jsd_per_dim = np.clip(
        0.5 * ((lp_sp - lm_sp).mean(0) + (lq_sq - lm_sq).mean(0)),
        0.0, np.log(2.0)
    )
    return float(jsd_per_dim.mean())


# ── Drift detectors ────────────────────────────────────────────────────────────

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
        self._delta_w = delta_warning
        self._delta_d = delta_drift
        self._warn_det  = ADWIN(delta=delta_warning)
        self._drift_det = ADWIN(delta=delta_drift)
        self._warning = self._drift = False

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
        self._warn_det  = ADWIN(delta=self._delta_w)
        self._drift_det = ADWIN(delta=self._delta_d)
        self._warning = self._drift = False


class DDMDetector(DriftDetectorBase):
    def __init__(self, threshold: float = MMD_THRESHOLD):
        self._threshold = threshold
        self._det = DDM()
        self._warning = self._drift = False

    def update(self, value: float) -> None:
        self._det.update(int(value > self._threshold))
        self._warning = self._det.warning_detected
        self._drift   = self._det.drift_detected

    @property
    def warning_detected(self) -> bool: return self._warning
    @property
    def drift_detected(self) -> bool:   return self._drift

    def reset(self) -> None:
        self._det = DDM()
        self._warning = self._drift = False


class EDDMDetector(DriftDetectorBase):
    def __init__(self, threshold: float = MMD_THRESHOLD):
        self._threshold = threshold
        self._det = EDDM()
        self._warning = self._drift = False

    def update(self, value: float) -> None:
        self._det.update(int(value > self._threshold))
        self._warning = self._det.warning_detected
        self._drift   = self._det.drift_detected

    @property
    def warning_detected(self) -> bool: return self._warning
    @property
    def drift_detected(self) -> bool:   return self._drift

    def reset(self) -> None:
        self._det = EDDM()
        self._warning = self._drift = False


def make_detector(name: str, metric: str) -> DriftDetectorBase:
    threshold = JSD_THRESHOLD if metric == "jsd" else MMD_THRESHOLD
    if name == "adwin": return ADWINDetector()
    if name == "ddm":   return DDMDetector(threshold=threshold)
    if name == "eddm":  return EDDMDetector(threshold=threshold)
    raise ValueError(f"Unknown detector: {name!r}")


# ── Model helpers ──────────────────────────────────────────────────────────────

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


def build_model(enc: dict, num_labels: int) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(
        enc["model_name"], trust_remote_code=enc["trust_remote_code"]
    )
    model = enc["model_class"].from_pretrained(
        enc["model_name"],
        num_labels=num_labels,
        trust_remote_code=enc["trust_remote_code"],
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=_detect_lora_targets(model),
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config).to(device)
    return model, tokenizer


def encode_and_predict(texts, model, tokenizer, batch_size=64):
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


def train_model(model, tokenizer, texts, labels, label_map,
                label: str = "train", epochs: int = BURNIN_EPOCHS):
    optimizer = AdamW(model.parameters(), lr=2e-4)
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(texts), TRAIN_BATCH):
            bt = texts[i : i + TRAIN_BATCH].tolist()
            bl = torch.tensor([label_map[l] for l in labels[i : i + TRAIN_BATCH]]).to(device)
            inputs = tokenizer(bt, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs, labels=bl).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print(f"  [{label.upper()}] {len(texts):,} samples × {epochs} epochs done")


# ── Sample selection strategies ────────────────────────────────────────────────

def _tsne_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D with t-SNE. Perplexity adapts to window size."""
    perplexity = min(30, max(5, len(embeddings) // 3 - 1))
    return TSNE(n_components=2, random_state=42,
                perplexity=perplexity, max_iter=300).fit_transform(embeddings)


def select_drift_samples(ref_np, win_np, X_win, y_win, win_2d) -> tuple:
    """Convex hull over the full window (original strategy from exp 13)."""
    n_win  = len(win_np)
    n_keep = max(ADAPT_MIN_SAMPLES, int(ADAPT_KEEP_RATIO * n_win))
    n_keep = min(n_keep, n_win)
    ref_centroid = ref_np.mean(axis=0)

    try:
        hull     = ConvexHull(win_2d)
        hull_idx = set(hull.vertices.tolist())
        method   = "convex_hull"
    except Exception:
        method   = "centroid_distance"
        hull_idx = set()

    if method == "convex_hull":
        non_hull = [i for i in range(n_win) if i not in hull_idx]
        if len(hull_idx) >= n_keep:
            hull_list = list(hull_idx)
            dists     = np.linalg.norm(win_np[hull_list] - ref_centroid, axis=1)
            selected  = np.array(hull_list)[np.argsort(dists)[::-1][:n_keep]]
        else:
            extra_needed = n_keep - len(hull_idx)
            non_dists    = np.linalg.norm(win_np[non_hull] - ref_centroid, axis=1)
            extra        = np.array(non_hull)[np.argsort(non_dists)[-extra_needed:]]
            selected     = np.array(list(hull_idx) + extra.tolist())
    else:
        dists    = np.linalg.norm(win_np - ref_centroid, axis=1)
        selected = np.argsort(dists)[-n_keep:]

    idx = np.sort(selected).astype(int)
    return X_win[idx], y_win[idx], idx


def select_drift_samples_per_class(ref_np, win_np, X_win, y_win, win_2d) -> tuple:
    """Convex hull applied per class on t-SNE 2D projection, then concatenated."""
    classes      = np.unique(y_win)
    ref_centroid = ref_np.mean(axis=0)
    selected_idx = []

    for cls in classes:
        cls_mask = np.where(y_win == cls)[0]
        cls_2d   = win_2d[cls_mask]
        cls_np   = win_np[cls_mask]

        n_keep_cls = max(int(ADAPT_KEEP_RATIO * len(cls_mask)),
                         ADAPT_MIN_SAMPLES // max(len(classes), 1))
        n_keep_cls = min(n_keep_cls, len(cls_mask))

        # not enough points for ConvexHull — keep all
        if len(cls_mask) < 4:
            selected_idx.extend(cls_mask.tolist())
            continue

        try:
            hull       = ConvexHull(cls_2d)
            hull_local = set(hull.vertices.tolist())
        except Exception:
            # fallback: distance to reference centroid within class
            dists        = np.linalg.norm(cls_np - ref_centroid, axis=1)
            chosen_local = np.argsort(dists)[-n_keep_cls:]
            selected_idx.extend(cls_mask[chosen_local].tolist())
            continue

        hull_global     = [cls_mask[i] for i in hull_local]
        non_hull_global = [cls_mask[i] for i in range(len(cls_mask))
                           if i not in hull_local]

        if len(hull_global) >= n_keep_cls:
            dists  = np.linalg.norm(win_np[hull_global] - ref_centroid, axis=1)
            chosen = np.array(hull_global)[np.argsort(dists)[::-1][:n_keep_cls]]
        else:
            extra_needed = n_keep_cls - len(hull_global)
            non_dists    = np.linalg.norm(win_np[non_hull_global] - ref_centroid, axis=1)
            extra        = np.array(non_hull_global)[np.argsort(non_dists)[-extra_needed:]]
            chosen       = np.array(hull_global + extra.tolist())

        selected_idx.extend(chosen.tolist())

    idx = np.sort(np.unique(selected_idx)).astype(int)
    return X_win[idx], y_win[idx], idx


# ── Plot ───────────────────────────────────────────────────────────────────────

def save_results_plot(
    window_positions, window_accuracies, window_entropies,
    mmd_scores, kl_scores, js_scores, centroid_scores,
    adapt_positions, warn_positions,
    ref_acc, ref_entropy, acc_plot_window,
    run_tag, base, subset, encoder_name,
    detector_name, metric_name, window_size, selection_method,
):
    drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]
    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(
        f"{base}-comdrift-{subset}  |  enc={encoder_name}  "
        f"sel={selection_method}  det={detector_name.upper()}  "
        f"metric={metric_name.upper()}  W={window_size}",
        fontsize=11,
    )

    def _fill(ax, ys, color, label, ref=None, ref_label=None, ylabel=""):
        ax.plot(window_positions, ys, color=color, linewidth=1.2, label=label)
        ax.fill_between(window_positions, ys, alpha=0.12, color=color)
        if ref is not None:
            ax.axhline(ref, color=color, linestyle=":", linewidth=1, alpha=0.7,
                       label=ref_label)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)

    _fill(axes[0], window_accuracies, "steelblue",
          f"Prequential acc (rolling {acc_plot_window})",
          ref=ref_acc, ref_label=f"Ref acc ({ref_acc:.3f})", ylabel="Accuracy")
    axes[0].set_ylim(0, 1.05)
    _fill(axes[1], window_entropies,  "seagreen", "Mean entropy",
          ref=ref_entropy, ref_label=f"Ref entropy ({ref_entropy:.4f})", ylabel="Entropy")
    _fill(axes[2], mmd_scores, "darkorange", "MMD",
          ref=MMD_THRESHOLD, ref_label=f"MMD thr ({MMD_THRESHOLD})", ylabel="MMD")
    _fill(axes[3], kl_scores,  "crimson",  "Sym KL div", ylabel="KL div")
    _fill(axes[4], js_scores,  "teal",     "JSD",
          ref=np.log(2), ref_label=f"Max JSD ({np.log(2):.3f})", ylabel="JSD (nats)")
    _fill(axes[5], centroid_scores, "mediumpurple", "Centroid dist", ylabel="Centroid dist")

    axes[5].set_xlabel("Position in stream", fontsize=10)

    for pos in warn_positions:
        for ax in axes:
            ax.axvline(x=pos, color="gold", linestyle=":", linewidth=1.0, alpha=0.7)
    for pos in adapt_positions:
        for ax in axes:
            ax.axvline(x=pos, color="limegreen", linestyle=":", linewidth=1.2, alpha=0.8)

    legend_patches = []
    for pos, col in zip(DRIFT_POSITIONS, drift_colors):
        for ax in axes:
            ax.axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
        legend_patches.append(mpatches.Patch(color=col, label=f"Drift @ {pos//1_000}k"))
    if adapt_positions:
        legend_patches.append(mpatches.Patch(color="limegreen", label="LoRA adapted"))
    if warn_positions:
        legend_patches.append(mpatches.Patch(color="gold", label="Warning"))

    x_ticks = np.arange(0, N_DRIFTED + 1, 25_000)
    axes[5].set_xticks(x_ticks)
    axes[5].set_xticklabels([f"{x//1_000}k" for x in x_ticks])
    axes[5].set_xlim(0, N_DRIFTED)
    axes[5].legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"{run_tag}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT]  → {fname}")


# ── Core run ───────────────────────────────────────────────────────────────────

def run_experiment(
    model, tokenizer, label_map, inv_label_map,
    burnin_texts, burnin_labels,
    stream_texts, stream_labels,
    window_size: int, n_ref_emb: int,
    detector_name: str, metric_name: str,
    selection_method: str,
    base: str, subset: int, encoder_name: str,
    run_tag: str,
) -> dict:

    acc_plot_window = window_size * ACC_PLOT_WINDOW_FACTOR
    n_windows       = len(stream_texts) // window_size

    ref_embs_t, ref_preds, ref_ent_arr = encode_and_predict(
        burnin_texts[:n_ref_emb], model, tokenizer
    )
    ref_np = ref_embs_t.cpu().numpy()

    ref_preds_orig  = np.array([inv_label_map[p] for p in ref_preds])
    ref_acc_initial = float((ref_preds_orig == burnin_labels[:n_ref_emb]).mean())
    ref_entropy_ini = float(ref_ent_arr.mean())
    ref_acc         = ref_acc_initial
    ref_entropy     = ref_entropy_ini

    gamma    = _estimate_gamma(ref_np)
    detector = make_detector(detector_name, metric_name)

    correct_history: list[int] = []
    warn_buffer_X:   list = []
    warn_buffer_y:   list = []

    window_positions  = []
    window_accuracies = []
    window_entropies  = []
    mmd_scores        = []
    kl_scores         = []
    js_scores         = []
    centroid_scores   = []
    adapt_positions   = []
    warn_positions    = []
    per_window_rows   = []

    n_warnings = 0
    n_drifts   = 0
    n_adapts   = 0
    first_drift_pos = None

    t_start = time.perf_counter()

    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size
        X_win = stream_texts[start:end]
        y_win = stream_labels[start:end]

        win_embs_t, win_preds, win_ent = encode_and_predict(X_win, model, tokenizer)
        win_np = win_embs_t.cpu().numpy()

        preds_orig = np.array([inv_label_map[p] for p in win_preds])
        correct_history.extend((preds_orig == y_win).astype(int).tolist())
        acc           = float(np.mean(correct_history[-acc_plot_window:]))
        entropy       = float(win_ent.mean())
        mmd           = compute_mmd(ref_np, win_np, gamma)
        kl            = compute_kl_divergence(ref_np, win_np)
        js            = compute_js_divergence(ref_np, win_np)
        centroid_dist = compute_centroid_distance(ref_np, win_np)
        pos           = BURNIN_SIZE + start + window_size // 2

        signal = mmd if metric_name == "mmd" else js
        detector.update(signal)

        warned  = detector.warning_detected
        drifted = detector.drift_detected
        adapted = False

        if warned:
            n_warnings += 1
            warn_positions.append(pos)
            if len(warn_buffer_X) < WARNING_BUFFER_MAX:
                warn_buffer_X.extend(X_win.tolist())
                warn_buffer_y.extend(y_win.tolist())

        if drifted:
            n_drifts += 1
            if first_drift_pos is None:
                first_drift_pos = pos
            print(f"  [DRIFT] w{i+1} pos={pos:,}  {metric_name.upper()}={signal:.4f}  "
                  f"buf={len(warn_buffer_X)}")

            # t-SNE projection for convex hull (perplexity adapts to window size)
            tsne_coords = _tsne_2d(win_np)

            if selection_method == "convex_hull_per_class":
                X_sel, y_sel, _ = select_drift_samples_per_class(
                    ref_np, win_np, X_win, y_win, tsne_coords
                )
            else:
                X_sel, y_sel, _ = select_drift_samples(
                    ref_np, win_np, X_win, y_win, tsne_coords
                )

            # combine hull-selected samples with warning buffer for training
            if warn_buffer_X:
                X_train = np.concatenate([np.array(warn_buffer_X), X_sel])
                y_train = np.concatenate([np.array(warn_buffer_y), y_sel])
            else:
                X_train, y_train = X_sel, y_sel
            train_model(model, tokenizer, X_train, y_train, label_map, label="adapt", epochs=ADAPT_EPOCHS)
            n_adapts += 1

            new_ref_t, new_ref_preds, new_ref_ent = encode_and_predict(
                X_win, model, tokenizer
            )
            ref_np      = new_ref_t.cpu().numpy()
            ref_entropy = float(new_ref_ent.mean())
            new_preds_orig = np.array([inv_label_map[p] for p in new_ref_preds])
            ref_acc     = float((new_preds_orig == y_win).mean())
            gamma       = _estimate_gamma(ref_np)
            detector.reset()
            warn_buffer_X = []
            warn_buffer_y = []
            adapt_positions.append(pos)
            adapted = True

        window_positions.append(pos)
        window_accuracies.append(acc)
        window_entropies.append(entropy)
        mmd_scores.append(mmd)
        kl_scores.append(kl)
        js_scores.append(js)
        centroid_scores.append(centroid_dist)

        per_window_rows.append({
            "window_idx": i + 1, "window_pos": pos,
            "prequential_acc": round(acc, 4),
            "entropy": round(entropy, 4),
            "mmd": round(mmd, 6), "kl": round(kl, 6),
            "jsd": round(js, 6), "centroid": round(centroid_dist, 6),
            "warning": int(warned), "drift": int(drifted), "adapted": int(adapted),
            "warning_buffer_size": len(warn_buffer_X),
        })

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{run_tag}] {i+1}/{n_windows}  "
                  f"acc={acc:.3f}  {metric_name}={signal:.4f}  "
                  f"elapsed={elapsed:.0f}s")

    wall_time = time.perf_counter() - t_start

    pw_path = os.path.join(PER_WINDOW_DIR, f"{run_tag}.csv")
    _save_per_window_csv(pw_path, per_window_rows)
    print(f"  [CSV]   → {pw_path}")

    save_results_plot(
        window_positions, window_accuracies, window_entropies,
        mmd_scores, kl_scores, js_scores, centroid_scores,
        adapt_positions, warn_positions,
        ref_acc_initial, ref_entropy_ini, acc_plot_window,
        run_tag, base, subset, encoder_name,
        detector_name, metric_name, window_size, selection_method,
    )

    post_drift_accs = [
        window_accuracies[j]
        for j, p in enumerate(window_positions)
        if p >= DRIFT_POSITIONS[0]
    ]

    return {
        "run_tag":             run_tag,
        "dataset":             f"{base}-{subset}-{DRIFT_TYPE}",
        "base":                base,
        "subset":              subset,
        "encoder":             encoder_name,
        "selection_method":    selection_method,
        "detector":            detector_name,
        "metric":              metric_name,
        "window_size":         window_size,
        "wall_time_s":         round(wall_time, 1),
        "n_windows":           n_windows,
        "n_warnings":          n_warnings,
        "n_drifts":            n_drifts,
        "n_adaptations":       n_adapts,
        "first_drift_pos":     first_drift_pos if first_drift_pos else "none",
        "mean_acc":            round(float(np.mean(window_accuracies)), 4),
        "mean_acc_post_drift": round(float(np.mean(post_drift_accs)), 4) if post_drift_accs else "n/a",
        "mean_mmd":            round(float(np.mean(mmd_scores)), 6),
        "mean_jsd":            round(float(np.mean(js_scores)), 6),
        "mean_kl":             round(float(np.mean(kl_scores)), 6),
        "mean_centroid":       round(float(np.mean(centroid_scores)), 6),
        "ref_acc_initial":     round(ref_acc_initial, 4),
        "ref_entropy_initial": round(ref_entropy_ini, 4),
    }


# ── Main sweep ─────────────────────────────────────────────────────────────────

_init_summary_csv()

total_runs = (len(DATASETS) * len(ENCODERS) * len(SWEEP_WINDOW_SIZES) *
              len(SWEEP_DETECTORS) * len(SWEEP_METRICS) * len(SWEEP_SELECTION_METHODS))
print(f"\nExperiment 15 sweep: {total_runs} total runs")
print(f"  datasets={len(DATASETS)}  encoders={len(ENCODERS)}")
print(f"  selection_methods={SWEEP_SELECTION_METHODS}")
print(f"  window_sizes={SWEEP_WINDOW_SIZES}")
print(f"  detectors={SWEEP_DETECTORS}  metrics={SWEEP_METRICS}\n")

completed = 0

for base, subset in DATASETS:
    com_path = f"{DATASET_DIR}/{base}-comdrift-{subset}-{DRIFT_TYPE}-ss.csv"
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

    label_map     = {v: i for i, v in enumerate(np.unique(all_labels))}
    inv_label_map = {i: v for v, i in label_map.items()}
    num_labels    = len(label_map)

    for enc in ENCODERS:
        print(f"\n{'='*60}")
        print(f"  Encoder: {enc['name']}")
        print(f"{'='*60}")

        for window_size in SWEEP_WINDOW_SIZES:
            n_ref_emb = window_size

            print(f"\n  [BURNIN] enc={enc['name']}  window={window_size}")
            model, tokenizer = build_model(enc, num_labels)
            model.print_trainable_parameters()
            train_model(model, tokenizer, burnin_texts, burnin_labels,
                        label_map, label="burnin")
            initial_state = copy.deepcopy(model.state_dict())

            for detector_name in SWEEP_DETECTORS:
                for metric_name in SWEEP_METRICS:
                    for sel_method in SWEEP_SELECTION_METHODS:
                        run_tag = (
                            f"{base}{subset}_{enc['name'].lower()}_"
                            f"{sel_method}_{detector_name}_{metric_name}_w{window_size}"
                        )

                        pw_path = os.path.join(PER_WINDOW_DIR, f"{run_tag}.csv")
                        if os.path.exists(pw_path):
                            print(f"  [SKIP]  {run_tag} (already done)")
                            completed += 1
                            continue

                        print(f"\n  ── Run {completed+1}/{total_runs}: {run_tag}")

                        model.load_state_dict(copy.deepcopy(initial_state))

                        summary = run_experiment(
                            model=model,
                            tokenizer=tokenizer,
                            label_map=label_map,
                            inv_label_map=inv_label_map,
                            burnin_texts=burnin_texts,
                            burnin_labels=burnin_labels,
                            stream_texts=stream_texts,
                            stream_labels=stream_labels,
                            window_size=window_size,
                            n_ref_emb=n_ref_emb,
                            detector_name=detector_name,
                            metric_name=metric_name,
                            selection_method=sel_method,
                            base=base,
                            subset=subset,
                            encoder_name=enc["name"],
                            run_tag=run_tag,
                        )

                        _append_summary(summary)
                        completed += 1
                        print(f"  [DONE]  {run_tag}  wall_time={summary['wall_time_s']}s  "
                              f"adaptations={summary['n_adaptations']}  "
                              f"({completed}/{total_runs} runs)")

            del model, tokenizer, initial_state
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

print(f"\n{'='*70}")
print(f"Sweep complete.  {completed}/{total_runs} runs.")
print(f"  Summary    → {SUMMARY_CSV}")
print(f"  Per-window → {PER_WINDOW_DIR}/")
print(f"  Plots      → {PLOTS_DIR}/")
