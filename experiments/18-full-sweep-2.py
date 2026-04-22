"""
Experiment 18 — Full Grid Sweep: Baseline vs Retrain (airbnb-2-2)

Datasets  : airbnb-2-2
Encoders  : ModernBERT, Jina-v2
Selection : convex_hull, convex_hull_per_class
Window    : 50, 100, 200
Detectors : ADWIN, KSWIN, Page-Hinkley   (all label-free, MMD/JSD-based)
Metrics   : MMD, JSD

Total retrain runs : 1 × 2 × 3 × 2 × 3 × 2 = 72
Baseline runs      : 1 × 2 × 3              =  6
                                              ────
Total passes       :                          78

Outputs:
  experiment18_results/{dataset}/plots/{tag}.png
  experiment18_results/{dataset}/per_window/{tag}.csv
  experiment18_results/summary.csv
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
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE

from river.drift import ADWIN, KSWIN

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
# Structure:
#   experiment17_results/
#     summary.csv
#     {dataset}/
#       plots/
#       per_window/
RESULTS_DIR = "experiment18_results_5"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _ds_dirs(dataset_name: str) -> tuple:
    """Return (plots_dir, per_window_dir) for a given dataset, creating them."""
    base       = os.path.join(RESULTS_DIR, dataset_name)
    plots_dir  = os.path.join(base, "plots")
    pw_dir     = os.path.join(base, "per_window")
    for d in [plots_dir, pw_dir]:
        os.makedirs(d, exist_ok=True)
    return plots_dir, pw_dir

# ── Fixed hyperparams ──────────────────────────────────────────────────────────
BURNIN_EPOCHS      = 3
ADAPT_EPOCHS       = 1
TRAIN_BATCH        = 16
PLOT_BUCKET_SIZE   = 1_000
ADAPT_KEEP_RATIO   = 0.5
ADAPT_MIN_SAMPLES  = 8
WARNING_BUFFER_MAX = 500
ACC_SMOOTHING      = 500  # fixed smoothing window for prequential accuracy,
                          # independent of detection window_size

# ── Sweep grid ─────────────────────────────────────────────────────────────────
SWEEP_WINDOW_SIZES      = [50, 100, 200]
SWEEP_SELECTION_METHODS = ["convex_hull", "convex_hull_per_class"]
SWEEP_METRICS           = ["mmd", "jsd"]
# Detectors are instantiated fresh per run — see DETECTOR_FACTORIES below

# ── Datasets ───────────────────────────────────────────────────────────────────
def _binarize_yelp_stars(star) -> int:
    """1,2 → 0 (negative)   3,4,5 → 1 (positive)"""
    return 0 if int(star) <= 2 else 1


_YELP_DRIFT_POS   = list(range(50_000, 600_000, 50_000))   # visual markers every 50k
_AIRBNB_DRIFT_POS = [50_000, 100_000, 150_000]

def _airbnb(subset: int, variant: int) -> dict:
    return {
        "name":            f"airbnb-{subset}-{variant}",
        "path":            f"datasets/airbnb-comdrift-{subset}-{variant}.csv",
        "text_col":        "review_treated",
        "label_col":       "label",
        "sort_col":        None,
        "n_total":         150_000,
        "drift_positions": _AIRBNB_DRIFT_POS,
        "burnin_size":     500,
    }

def _airbnb_ss(subset: int) -> dict:
    return {
        "name":            f"airbnb-{subset}-ss",
        "path":            f"datasets/airbnb-comdrift-{subset}-1-ss.csv",
        "text_col":        "review_treated",
        "label_col":       "label",
        "sort_col":        None,
        "n_total":         150_000,
        "drift_positions": _AIRBNB_DRIFT_POS,
        "burnin_size":     500,
    }

def _yelp(subset: int, variant: int) -> dict:
    return {
        "name":            f"yelp-{subset}-{variant}",
        "path":            f"datasets/yelp-comdrift-{subset}-{variant}.csv",
        "text_col":        "text",
        "label_col":       "stars",
        "label_transform": _binarize_yelp_stars,
        "sort_col":        "year_review",
        "n_total":         None,
        "drift_positions": _YELP_DRIFT_POS,
        "burnin_size":     500,
    }

def _yelp_ss(subset: int) -> dict:
    return {
        "name":            f"yelp-{subset}-ss",
        "path":            f"datasets/yelp-comdrift-{subset}-1-ss.csv",
        "text_col":        "text",
        "label_col":       "stars",
        "label_transform": _binarize_yelp_stars,
        "sort_col":        "year_review",
        "n_total":         None,
        "drift_positions": _YELP_DRIFT_POS,
        "burnin_size":     500,
    }

DATASETS = [_airbnb(2, 2)]

# ── Encoders ───────────────────────────────────────────────────────────────────
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
    "run_tag", "dataset", "encoder",
    "selection_method", "detector", "metric", "window_size",
    "mean_acc_baseline", "mean_acc_retrain",
    "mean_acc_post_drift_baseline", "mean_acc_post_drift_retrain",
    "n_adaptations", "wall_time_s",
]

PER_WINDOW_FIELDS = [
    "window_idx", "window_pos",
    "prequential_acc", "mmd", "jsd",
    "warning", "drift", "adapted",
]


def _init_summary_csv() -> None:
    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()


def _append_summary(row: dict) -> None:
    with open(SUMMARY_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow(row)


def _save_per_window(path: str, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_WINDOW_FIELDS)
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


def compute_jsd(P: np.ndarray, Q: np.ndarray,
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
        0.0, np.log(2.0),
    )
    return float(jsd_per_dim.mean())


# ── Drift detectors (all label-free, window-level MMD or JSD) ─────────────────

class ADWINDetector:
    """Two ADWIN instances on the signal stream — one for warning, one for drift."""
    def __init__(self, delta_warning: float = 0.3, delta_drift: float = 0.15):
        self._delta_w   = delta_warning
        self._delta_d   = delta_drift
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
    def drift_detected(self)   -> bool: return self._drift

    def reset(self) -> None:
        self._warn_det  = ADWIN(delta=self._delta_w)
        self._drift_det = ADWIN(delta=self._delta_d)
        self._warning = self._drift = False


class KSWINDetector:
    """Two KSWIN instances on the signal stream — one for warning, one for drift.
    Higher alpha → rejects H0 more easily → fires earlier (warning).
    Lower alpha → more conservative → fires later (drift)."""
    def __init__(self, alpha_warning: float = 0.05, alpha_drift: float = 0.01,
                 window_size: int = 100, stat_size: int = 30, seed: int = 42):
        self._alpha_w     = alpha_warning
        self._alpha_d     = alpha_drift
        self._window_size = window_size
        self._stat_size   = stat_size
        self._seed        = seed
        self._warn_det    = KSWIN(alpha=alpha_warning, window_size=window_size,
                                  stat_size=stat_size, seed=seed)
        self._drift_det   = KSWIN(alpha=alpha_drift,   window_size=window_size,
                                  stat_size=stat_size, seed=seed)
        self._warning = self._drift = False

    def update(self, value: float) -> None:
        self._warn_det.update(value)
        self._drift_det.update(value)
        self._drift   = self._drift_det.drift_detected
        self._warning = self._warn_det.drift_detected and not self._drift

    @property
    def warning_detected(self) -> bool: return self._warning
    @property
    def drift_detected(self)   -> bool: return self._drift

    def reset(self) -> None:
        self._warn_det  = KSWIN(alpha=self._alpha_w, window_size=self._window_size,
                                stat_size=self._stat_size, seed=self._seed)
        self._drift_det = KSWIN(alpha=self._alpha_d, window_size=self._window_size,
                                stat_size=self._stat_size, seed=self._seed)
        self._warning = self._drift = False


class PageHinkleyDetector:
    """Page-Hinkley test with warmup-frozen baseline mean."""
    def __init__(self, warmup_windows: int = 20, delta: float = 0.001,
                 lambda_: float = 1.0):
        self._warmup  = warmup_windows
        self._delta   = delta
        self._lambda  = lambda_
        self._history : list[float] = []
        self._mean    = 0.0
        self._ready   = False
        self._sum     = 0.0
        self._min_sum = float("inf")
        self._warning = self._drift = False

    def update(self, value: float) -> None:
        if not self._ready:
            self._history.append(value)
            if len(self._history) >= self._warmup:
                self._mean  = float(np.mean(self._history))
                self._ready = True
            self._warning = self._drift = False
            return
        self._sum     += (value - self._mean - self._delta)
        self._min_sum  = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        self._warning = ph > self._lambda * 0.5
        self._drift   = ph > self._lambda

    @property
    def warning_detected(self) -> bool: return self._warning
    @property
    def drift_detected(self)   -> bool: return self._drift

    def reset(self) -> None:
        self._sum     = 0.0
        self._min_sum = float("inf")
        self._warning = self._drift = False  # keep frozen mean after reset


DETECTOR_FACTORIES = {
    "KSWIN": lambda: KSWINDetector(),
}

DETECTOR_STYLES = {
    "ADWIN": {"color": "darkorange",  "linestyle": "-"},
    "KSWIN": {"color": "seagreen",    "linestyle": "-"},
    "PH":    {"color": "mediumpurple","linestyle": "-"},
}

# ── Model helpers ──────────────────────────────────────────────────────────────

def _detect_lora_targets(model: torch.nn.Module) -> list:
    candidates = {"query", "key", "value", "q_proj", "k_proj", "v_proj",
                  "Wqkv", "q_lin", "v_lin"}
    found: set = set()
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
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=_detect_lora_targets(model),
        task_type=TaskType.SEQ_CLS,
    )
    torch.manual_seed(42)  # seed LoRA A (kaiming_uniform) init
    model = get_peft_model(model, lora_cfg).to(device)
    return model, tokenizer


def encode_and_predict(texts, model, tokenizer, batch_size: int = 64):
    all_emb, all_preds = [], []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        all_emb.append(out.hidden_states[-1][:, 0, :].cpu())
        all_preds.append(out.logits.argmax(dim=-1).cpu().numpy())
    return torch.cat(all_emb, dim=0), np.concatenate(all_preds)


def train_model(model, tokenizer, texts, labels, label_map,
                label: str = "train", epochs: int = BURNIN_EPOCHS,
                lr: float = 2e-4) -> None:
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for i in range(0, len(texts), TRAIN_BATCH):
            bt = texts[i : i + TRAIN_BATCH].tolist()
            bl = torch.tensor(
                [label_map[l] for l in labels[i : i + TRAIN_BATCH]]
            ).to(device)
            inputs = tokenizer(bt, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs, labels=bl).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print(f"  [{label.upper()}] {len(texts):,} samples × {epochs} epochs done")


# ── Sample selection ───────────────────────────────────────────────────────────

def _tsne_2d(embeddings: np.ndarray) -> np.ndarray:
    perplexity = min(30, max(5, len(embeddings) // 3 - 1))
    return TSNE(n_components=2, random_state=42,
                perplexity=perplexity, max_iter=300).fit_transform(embeddings)


def select_convex_hull(ref_np, win_np, X_win, y_win, win_2d) -> tuple:
    """Convex hull over the full window (all classes together)."""
    ref_centroid = ref_np.mean(axis=0)
    n_keep = max(int(ADAPT_KEEP_RATIO * len(win_np)), ADAPT_MIN_SAMPLES)
    n_keep = min(n_keep, len(win_np))

    if len(win_np) < 4:
        return X_win, y_win, np.arange(len(win_np))

    try:
        hull      = ConvexHull(win_2d)
        hull_idx  = list(hull.vertices)
    except Exception:
        dists  = np.linalg.norm(win_np - ref_centroid, axis=1)
        chosen = np.argsort(dists)[-n_keep:]
        return X_win[chosen], y_win[chosen], chosen

    non_hull = [i for i in range(len(win_np)) if i not in set(hull_idx)]

    if len(hull_idx) >= n_keep:
        dists  = np.linalg.norm(win_np[hull_idx] - ref_centroid, axis=1)
        chosen = np.array(hull_idx)[np.argsort(dists)[::-1][:n_keep]]
    else:
        extra_needed = n_keep - len(hull_idx)
        non_dists = np.linalg.norm(win_np[non_hull] - ref_centroid, axis=1)
        extra     = np.array(non_hull)[np.argsort(non_dists)[-extra_needed:]]
        chosen    = np.array(hull_idx + extra.tolist())

    idx = np.sort(np.unique(chosen)).astype(int)
    return X_win[idx], y_win[idx], idx


def select_convex_hull_per_class(ref_np, win_np, X_win, y_win, win_2d) -> tuple:
    """Convex hull applied per class, then concatenated."""
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

        if len(cls_mask) < 4:
            selected_idx.extend(cls_mask.tolist())
            continue

        try:
            hull       = ConvexHull(cls_2d)
            hull_local = set(hull.vertices.tolist())
        except Exception:
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


SELECT_FN = {
    "convex_hull":           select_convex_hull,
    "convex_hull_per_class": select_convex_hull_per_class,
}

# ── Stream passes ──────────────────────────────────────────────────────────────

def pass_baseline(model, tokenizer, inv_label_map,
                  stream_texts, stream_labels,
                  window_size: int, burnin_size: int) -> tuple:
    """No drift detection, no retraining. Returns (positions, accuracies)."""
    n_windows       = len(stream_texts) // window_size
    correct_history = []
    positions, accuracies = [], []

    for i in range(n_windows):
        start = i * window_size
        X_win = stream_texts[start : start + window_size]
        y_win = stream_labels[start : start + window_size]

        _, win_preds = encode_and_predict(X_win, model, tokenizer)
        preds_orig = np.array([inv_label_map[p] for p in win_preds])
        correct_history.extend((preds_orig == y_win).astype(int).tolist())
        acc = float(np.mean(correct_history[-ACC_SMOOTHING:]))
        pos = burnin_size + start + window_size // 2

        positions.append(pos)
        accuracies.append(acc)

    return positions, accuracies


def pass_retrain(model, tokenizer, label_map, inv_label_map,
                 stream_texts, stream_labels,
                 burnin_texts, burnin_labels,
                 window_size: int, burnin_size: int,
                 detector, det_name: str,
                 metric_name: str,
                 selection_fn) -> tuple:
    """Drift detection + LoRA adaptation.
    Returns (positions, accuracies, adapt_positions, per_window_rows)."""
    n_windows       = len(stream_texts) // window_size
    correct_history = []
    positions, accuracies, adapt_positions = [], [], []
    warn_buffer_X, warn_buffer_y = [], []
    per_window_rows = []

    ref_embs_t, _ = encode_and_predict(burnin_texts, model, tokenizer)
    ref_np = ref_embs_t.cpu().numpy()
    gamma  = _estimate_gamma(ref_np)

    for i in range(n_windows):
        start = i * window_size
        X_win = stream_texts[start : start + window_size]
        y_win = stream_labels[start : start + window_size]

        win_embs_t, win_preds = encode_and_predict(X_win, model, tokenizer)
        win_np = win_embs_t.cpu().numpy()

        preds_orig = np.array([inv_label_map[p] for p in win_preds])
        correct_history.extend((preds_orig == y_win).astype(int).tolist())
        acc = float(np.mean(correct_history[-ACC_SMOOTHING:]))
        pos = burnin_size + start + window_size // 2

        mmd = compute_mmd(ref_np, win_np, gamma)
        jsd = compute_jsd(ref_np, win_np)

        signal = mmd if metric_name == "mmd" else jsd
        detector.update(signal)

        warned  = detector.warning_detected
        drifted = detector.drift_detected
        adapted = False

        if warned and not drifted:
            if len(warn_buffer_X) < WARNING_BUFFER_MAX:
                warn_buffer_X.extend(X_win.tolist())
                warn_buffer_y.extend(y_win.tolist())
            print(f"  [WARN/{det_name}/{metric_name}] w{i+1}  {metric_name.upper()}={signal:.4f}"
                  f"  buf={len(warn_buffer_X)}")

        if drifted:
            print(f"  [DRIFT/{det_name}/{metric_name}] w{i+1} pos={pos:,}"
                  f"  {metric_name.upper()}={signal:.4f}  buf={len(warn_buffer_X)}")

            tsne_coords = _tsne_2d(win_np)
            X_sel, y_sel, _ = selection_fn(ref_np, win_np, X_win, y_win, tsne_coords)

            # combine with warning buffer
            if warn_buffer_X:
                X_new = np.concatenate([np.array(warn_buffer_X), X_sel])
                y_new = np.concatenate([np.array(warn_buffer_y), y_sel])
            else:
                X_new, y_new = X_sel, y_sel

            # experience replay: 50% burn-in to prevent catastrophic forgetting
            n_replay   = len(X_new) // 2
            replay_idx = np.random.choice(len(burnin_texts), n_replay, replace=False)
            X_train    = np.concatenate([X_new, burnin_texts[replay_idx]])
            y_train    = np.concatenate([y_new, burnin_labels[replay_idx]])
            perm       = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]

            train_model(model, tokenizer, X_train, y_train, label_map,
                        label="adapt", epochs=ADAPT_EPOCHS, lr=5e-5)

            new_ref_t, _ = encode_and_predict(X_win, model, tokenizer)
            ref_np = new_ref_t.cpu().numpy()
            gamma  = _estimate_gamma(ref_np)
            detector.reset()
            warn_buffer_X, warn_buffer_y = [], []
            adapt_positions.append(pos)
            adapted = True

        positions.append(pos)
        accuracies.append(acc)
        per_window_rows.append({
            "window_idx":      i + 1,
            "window_pos":      pos,
            "prequential_acc": round(acc, 4),
            "mmd":             round(mmd, 6),
            "jsd":             round(jsd, 6),
            "warning":         int(warned),
            "drift":           int(drifted),
            "adapted":         int(adapted),
        })

    return positions, accuracies, adapt_positions, per_window_rows


# ── Plot ───────────────────────────────────────────────────────────────────────

def _bucket(positions, values, bucket_size: int = PLOT_BUCKET_SIZE):
    buckets: dict = {}
    for pos, val in zip(positions, values):
        key = (pos // bucket_size) * bucket_size
        buckets.setdefault(key, []).append(val)
    keys = sorted(buckets)
    return (
        [k + bucket_size // 2 for k in keys],
        [float(np.mean(buckets[k])) for k in keys],
    )


def save_plot(positions_base, acc_base,
              retrain_results: dict,
              run_tag: str, dataset_name: str, encoder_name: str,
              sel_method: str, metric_name: str, window_size: int,
              drift_positions: list, n_total: int,
              plots_dir: str = RESULTS_DIR) -> None:
    drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"{dataset_name}  enc={encoder_name}  sel={sel_method}"
        f"  metric={metric_name.upper()}  W={window_size}",
        fontsize=10,
    )

    pb, ab = _bucket(positions_base, acc_base)
    ax.plot(pb, ab, color="steelblue", linewidth=1.8,
            label=f"Baseline (mean={np.mean(acc_base):.3f})")
    ax.fill_between(pb, ab, alpha=0.08, color="steelblue")

    for det_name, (pos_r, acc_r, adapt_pos, _) in retrain_results.items():
        style  = DETECTOR_STYLES.get(det_name, {"color": "gray", "linestyle": "--"})
        pr, ar = _bucket(pos_r, acc_r)
        ax.plot(pr, ar, linewidth=1.5,
                label=f"{det_name} (mean={np.mean(acc_r):.3f})", **style)
        for ap in adapt_pos:
            ax.axvline(x=ap, color=style["color"], linestyle=":",
                       linewidth=1.0, alpha=0.6)

    for dp, col in zip(drift_positions, drift_colors):
        ax.axvline(x=dp, color=col, linestyle="--", linewidth=1.8, alpha=0.85)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Prequential Accuracy", fontsize=9)
    ax.set_xlabel("Stream position", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    bucket_step = max(n_total // 10, 1_000)
    x_ticks = np.arange(0, n_total + 1, bucket_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x//1_000}k" for x in x_ticks], fontsize=7)
    ax.set_xlim(0, n_total)

    plt.tight_layout()
    fname = os.path.join(plots_dir, f"{run_tag}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] → {fname}")


# ── Main ───────────────────────────────────────────────────────────────────────

_init_summary_csv()

total_combos = (len(DATASETS) * len(ENCODERS) * len(SWEEP_WINDOW_SIZES)
                * len(SWEEP_SELECTION_METHODS) * len(SWEEP_METRICS)
                * len(DETECTOR_FACTORIES))
print(f"\nExperiment 18 — Full Grid Sweep (airbnb-2-2)")
print(f"  {len(DATASETS)} datasets × {len(ENCODERS)} encoders"
      f" × {len(SWEEP_WINDOW_SIZES)} windows"
      f" × {len(SWEEP_SELECTION_METHODS)} selections"
      f" × {len(DETECTOR_FACTORIES)} detectors"
      f" × {len(SWEEP_METRICS)} metrics"
      f" = {total_combos} retrain runs\n")

completed = 0

for ds in DATASETS:
    dataset_name    = ds["name"]
    drift_positions = ds["drift_positions"]
    burnin_size     = ds["burnin_size"]
    text_col        = ds["text_col"]

    print(f"\n{'#'*70}")
    print(f"  Dataset: {ds['path']}")
    print(f"{'#'*70}")

    if not os.path.exists(ds["path"]):
        print(f"  [SKIP] file not found: {ds['path']}")
        continue

    com_df = pd.read_csv(ds["path"]).dropna(subset=[text_col])
    if ds["sort_col"] and ds["sort_col"] in com_df.columns:
        com_df = com_df.sort_values(ds["sort_col"]).reset_index(drop=True)
    n_total    = ds["n_total"]   # None means use all rows
    all_texts  = com_df[text_col].astype(str).values[:n_total]
    raw_labels = com_df[ds.get("label_col", "label")].values[:n_total]
    transform  = ds.get("label_transform", None)
    all_labels = np.array([transform(l) for l in raw_labels]) if transform else raw_labels

    label_map     = {v: i for i, v in enumerate(np.unique(all_labels))}
    inv_label_map = {i: v for v, i in label_map.items()}
    num_labels    = len(label_map)

    burnin_texts  = all_texts[:burnin_size]
    burnin_labels = all_labels[:burnin_size]
    stream_texts  = all_texts[burnin_size:]
    stream_labels = all_labels[burnin_size:]

    plots_dir, pw_dir = _ds_dirs(dataset_name)

    for enc in ENCODERS:
        enc_tag = enc["name"]
        print(f"\n{'='*60}")
        print(f"  Encoder: {enc_tag}")
        print(f"{'='*60}")

        t_enc_start = time.perf_counter()
        model, tokenizer = build_model(enc, num_labels)
        model.print_trainable_parameters()
        train_model(model, tokenizer, burnin_texts, burnin_labels,
                    label_map, label="burnin")
        initial_state = copy.deepcopy(model.state_dict())

        for window_size in SWEEP_WINDOW_SIZES:

            # ── baseline (once per window_size) ──────────────────────────
            base_tag  = f"{enc_tag.lower()}_baseline_w{window_size}"
            base_done = os.path.join(pw_dir, f"{base_tag}.csv")

            if not os.path.exists(base_done):
                print(f"\n  [BASELINE] W={window_size} ...")
                model.load_state_dict(copy.deepcopy(initial_state))
                pos_base, acc_base = pass_baseline(
                    model, tokenizer, inv_label_map,
                    stream_texts, stream_labels,
                    window_size=window_size, burnin_size=burnin_size,
                )
                _save_per_window(base_done, [
                    {"window_idx": i+1, "window_pos": p,
                     "prequential_acc": round(a, 4),
                     "mmd": "", "jsd": "",
                     "warning": 0, "drift": 0, "adapted": 0}
                    for i, (p, a) in enumerate(zip(pos_base, acc_base))
                ])
            else:
                # reload from disk
                base_df  = pd.read_csv(base_done)
                pos_base = base_df["window_pos"].tolist()
                acc_base = base_df["prequential_acc"].tolist()

            post_drift_base = [a for p, a in zip(pos_base, acc_base)
                               if p >= drift_positions[0]]

            # ── retrain grid ─────────────────────────────────────────────
            for sel_method in SWEEP_SELECTION_METHODS:
                for metric_name in SWEEP_METRICS:

                    retrain_results = {}
                    for det_name, det_factory in DETECTOR_FACTORIES.items():

                        run_tag = (f"{enc_tag.lower()}"
                                   f"_{sel_method}_{det_name.lower()}"
                                   f"_{metric_name}_w{window_size}")
                        pw_path = os.path.join(pw_dir, f"{run_tag}.csv")

                        if os.path.exists(pw_path):
                            print(f"  [SKIP] {run_tag}")
                            df_r = pd.read_csv(pw_path)
                            pos_r = df_r["window_pos"].tolist()
                            acc_r = df_r["prequential_acc"].tolist()
                            adapt_pos = df_r.loc[df_r["adapted"] == 1,
                                                 "window_pos"].tolist()
                            retrain_results[det_name] = (pos_r, acc_r,
                                                          adapt_pos, [])
                            completed += 1
                            continue

                        t_run = time.perf_counter()
                        print(f"\n  [{completed+1}] {run_tag}")
                        model.load_state_dict(copy.deepcopy(initial_state))

                        pos_r, acc_r, adapt_pos, pw_rows = pass_retrain(
                            model, tokenizer, label_map, inv_label_map,
                            stream_texts, stream_labels,
                            burnin_texts, burnin_labels,
                            window_size=window_size,
                            burnin_size=burnin_size,
                            detector=det_factory(),
                            det_name=det_name,
                            metric_name=metric_name,
                            selection_fn=SELECT_FN[sel_method],
                        )
                        retrain_results[det_name] = (pos_r, acc_r,
                                                      adapt_pos, pw_rows)
                        _save_per_window(pw_path, pw_rows)

                        post_drift_ret = [a for p, a in zip(pos_r, acc_r)
                                         if p >= drift_positions[0]]
                        wall_time = round(time.perf_counter() - t_run, 1)

                        _append_summary({
                            "run_tag":        f"{dataset_name}_{run_tag}",
                            "dataset":        dataset_name,
                            "encoder":        enc_tag,
                            "selection_method": sel_method,
                            "detector":       det_name,
                            "metric":         metric_name,
                            "window_size":    window_size,
                            "mean_acc_baseline":
                                round(float(np.mean(acc_base)), 4),
                            "mean_acc_retrain":
                                round(float(np.mean(acc_r)), 4),
                            "mean_acc_post_drift_baseline":
                                round(float(np.mean(post_drift_base)), 4)
                                if post_drift_base else "n/a",
                            "mean_acc_post_drift_retrain":
                                round(float(np.mean(post_drift_ret)), 4)
                                if post_drift_ret else "n/a",
                            "n_adaptations":  len(adapt_pos),
                            "wall_time_s":    wall_time,
                        })
                        print(f"  [DONE] {run_tag}  n_adapt={len(adapt_pos)}"
                              f"  wall={wall_time}s")
                        completed += 1

                    # ── plot: baseline + 3 detectors for this (sel, metric, W)
                    plot_tag  = (f"{enc_tag.lower()}"
                                 f"_{sel_method}_{metric_name}_w{window_size}")
                    plot_path = os.path.join(plots_dir, f"{plot_tag}.png")
                    if not os.path.exists(plot_path) and retrain_results:
                        save_plot(
                            pos_base, acc_base,
                            retrain_results,
                            plot_tag,
                            f"{dataset_name}  {enc_tag}",
                            enc_tag,
                            sel_method, metric_name, window_size,
                            drift_positions, len(all_texts),
                            plots_dir=plots_dir,
                        )

        wall_enc = round(time.perf_counter() - t_enc_start, 1)
        print(f"\n  [ENC DONE] {enc_tag}  total={wall_enc}s")
        del model, tokenizer, initial_state
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

print(f"\n{'='*70}")
print(f"Experiment 18 complete.  {completed} runs.")
print(f"  Summary → {SUMMARY_CSV}")
print(f"  Plots   → {RESULTS_DIR}/{{dataset}}/plots/")
