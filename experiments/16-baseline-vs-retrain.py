"""
Experiment 16 — Baseline vs Retrain comparison

For each (dataset, encoder): runs two passes over the same stream —
  1. baseline  : no drift detection, no LoRA retraining (from exp 14)
  2. retrain   : drift detection + LoRA adaptation (from exp 15)

Plots both accuracy curves on the same panel, smoothed to 1k-sample
buckets, with true drift positions marked as dashed vertical lines.

Fixed retrain config:
  window_size      : 100
  detector         : adwin
  metric           : mmd
  selection_method : convex_hull_per_class

Outputs:
  experiment16_results/plots/{tag}.png     ← one plot per (dataset, encoder)
  experiment16_results/summary.csv         ← mean accuracy for each run
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
RESULTS_DIR = "experiment16_results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")

for d in [RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparams ────────────────────────────────────────────────────────────────
DATASET_DIR = "datasets"

BURNIN_SIZE     = 1_000
N_DRIFTED       = 22_000        # total samples to use from tech dataset
DRIFT_POSITIONS = [11_600]      # approximate 2009→2026 boundary in stream
TEXT_COL        = "text"
SORT_BY_DATE    = True          # sort dataset by created_at before streaming

WINDOW_SIZE      = 200
BURNIN_EPOCHS    = 3
ADAPT_EPOCHS     = 1
TRAIN_BATCH      = 16
PLOT_BUCKET_SIZE = 1_000

# fixed retrain config — threshold-based detection (simpler, works with short streams)
DETECTOR_NAME    = "threshold"
METRIC_NAME      = "mmd"
SELECTION_METHOD = "convex_hull_per_class"
MMD_THRESHOLD    = 0.02
WARNING_BUFFER_MAX = 500
ADAPT_KEEP_RATIO   = 0.5
ADAPT_MIN_SAMPLES  = 8

# ── Dataset / encoder lists ────────────────────────────────────────────────────
DATASETS = [
    "tech_non_tech_dataset.csv",
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
    "run_tag", "dataset", "base", "subset", "encoder",
    "mean_acc_baseline", "mean_acc_retrain",
    "mean_acc_post_drift_baseline", "mean_acc_post_drift_retrain",
    "n_adaptations", "wall_time_s",
]


def _init_summary_csv() -> None:
    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()


def _append_summary(row: dict) -> None:
    with open(SUMMARY_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow(row)


# ── Metric / model helpers ─────────────────────────────────────────────────────

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
    all_emb, all_preds = [], []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        all_emb.append(outputs.hidden_states[-1][:, 0, :].cpu())
        all_preds.append(outputs.logits.argmax(dim=-1).cpu().numpy())
    return (
        torch.cat(all_emb, dim=0),
        np.concatenate(all_preds),
    )


def train_model(model, tokenizer, texts, labels, label_map,
                label: str = "train", epochs: int = BURNIN_EPOCHS, lr: float = 2e-4):
    optimizer = AdamW(model.parameters(), lr=lr)
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


# ── Sample selection (convex hull per class, t-SNE) ───────────────────────────

def _tsne_2d(embeddings: np.ndarray) -> np.ndarray:
    perplexity = min(30, max(5, len(embeddings) // 3 - 1))
    return TSNE(n_components=2, random_state=42,
                perplexity=perplexity, max_iter=300).fit_transform(embeddings)


def select_drift_samples_per_class(ref_np, win_np, X_win, y_win, win_2d) -> tuple:
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


# ── ADWIN detector ─────────────────────────────────────────────────────────────

class ADWINDetector:
    def __init__(self, delta_warning: float = 0.01, delta_drift: float = 0.001):
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
    def drift_detected(self) -> bool:   return self._drift

    def reset(self) -> None:
        self._warn_det  = ADWIN(delta=self._delta_w)
        self._drift_det = ADWIN(delta=self._delta_d)
        self._warning = self._drift = False


# ── Plot ───────────────────────────────────────────────────────────────────────

def _bucket(positions, values):
    buckets: dict[int, list] = {}
    for pos, val in zip(positions, values):
        key = (pos // PLOT_BUCKET_SIZE) * PLOT_BUCKET_SIZE
        buckets.setdefault(key, []).append(val)
    keys = sorted(buckets)
    return (
        [k + PLOT_BUCKET_SIZE // 2 for k in keys],
        [float(np.mean(buckets[k])) for k in keys],
    )


def save_comparison_plot(
    positions_base, acc_base,
    positions_ret,  acc_ret,
    adapt_positions,
    run_tag, base, subset, encoder_name,
):
    pos_b, acc_b_s = _bucket(positions_base, acc_base)
    pos_r, acc_r_s = _bucket(positions_ret,  acc_ret)

    drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"{base}-comdrift-{subset}  |  enc={encoder_name}  "
        f"[Baseline vs Retrain — {SELECTION_METHOD}, {DETECTOR_NAME.upper()}, "
        f"{METRIC_NAME.upper()}, W={WINDOW_SIZE}]",
        fontsize=11,
    )

    ax.plot(pos_b, acc_b_s, color="steelblue",  linewidth=1.5, label="Baseline (no retrain)")
    ax.plot(pos_r, acc_r_s, color="darkorange",  linewidth=1.5, label="With LoRA retrain")
    ax.fill_between(pos_b, acc_b_s, alpha=0.10, color="steelblue")
    ax.fill_between(pos_r, acc_r_s, alpha=0.10, color="darkorange")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlabel("Position in stream", fontsize=10)
    ax.grid(alpha=0.25)

    legend_patches = []
    for pos, col in zip(DRIFT_POSITIONS, drift_colors):
        ax.axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
        legend_patches.append(mpatches.Patch(color=col, label=f"Drift @ {pos//1_000}k"))
    for pos in adapt_positions:
        ax.axvline(x=pos, color="limegreen", linestyle=":", linewidth=1.2, alpha=0.8)
    if adapt_positions:
        legend_patches.append(mpatches.Patch(color="limegreen", label="LoRA adapted"))

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="steelblue",  linewidth=2, label="Baseline"),
            plt.Line2D([0], [0], color="darkorange",  linewidth=2, label="With retrain"),
            *legend_patches,
        ],
        loc="lower left", fontsize=9,
    )

    x_ticks = np.arange(0, N_DRIFTED + 1, 25_000)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x//1_000}k" for x in x_ticks])
    ax.set_xlim(0, N_DRIFTED)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"{run_tag}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT]  → {fname}")


# ── Stream passes ──────────────────────────────────────────────────────────────

def pass_baseline(model, tokenizer, inv_label_map, stream_texts, stream_labels):
    """One pass with no drift detection or retraining. Returns (positions, accuracies)."""
    n_windows       = len(stream_texts) // WINDOW_SIZE
    correct_history = []
    positions       = []
    accuracies      = []

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        X_win = stream_texts[start : start + WINDOW_SIZE]
        y_win = stream_labels[start : start + WINDOW_SIZE]

        _, win_preds = encode_and_predict(X_win, model, tokenizer)
        preds_orig = np.array([inv_label_map[p] for p in win_preds])
        correct_history.extend((preds_orig == y_win).astype(int).tolist())
        acc = float(np.mean(correct_history[-WINDOW_SIZE * 4:]))
        pos = BURNIN_SIZE + start + WINDOW_SIZE // 2

        positions.append(pos)
        accuracies.append(acc)

    return positions, accuracies


def pass_retrain(model, tokenizer, label_map, inv_label_map,
                 stream_texts, stream_labels):
    """One pass with ADWIN + MMD drift detection and LoRA adaptation.
    Returns (positions, accuracies, adapt_positions)."""
    n_windows       = len(stream_texts) // WINDOW_SIZE
    correct_history = []
    positions       = []
    accuracies      = []
    adapt_positions = []
    warn_buffer_X   = []
    warn_buffer_y   = []

    # reference embeddings from burnin (same distribution the model was trained on)
    ref_embs_t, _ = encode_and_predict(burnin_texts, model, tokenizer)
    ref_np = ref_embs_t.cpu().numpy()
    gamma  = _estimate_gamma(ref_np)
    det    = ADWINDetector(delta_warning=0.05, delta_drift=0.01)

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        X_win = stream_texts[start : start + WINDOW_SIZE]
        y_win = stream_labels[start : start + WINDOW_SIZE]

        win_embs_t, win_preds = encode_and_predict(X_win, model, tokenizer)
        win_np = win_embs_t.cpu().numpy()

        preds_orig = np.array([inv_label_map[p] for p in win_preds])
        correct_history.extend((preds_orig == y_win).astype(int).tolist())
        acc = float(np.mean(correct_history[-WINDOW_SIZE * 4:]))
        pos = BURNIN_SIZE + start + WINDOW_SIZE // 2

        mmd = compute_mmd(ref_np, win_np, gamma)
        det.update(mmd)

        if det.warning_detected:
            if len(warn_buffer_X) < WARNING_BUFFER_MAX:
                warn_buffer_X.extend(X_win.tolist())
                warn_buffer_y.extend(y_win.tolist())
            print(f"  [WARN]  w{i+1} pos={pos:,}  MMD={mmd:.4f}  buf={len(warn_buffer_X)}")

        if det.drift_detected:
            print(f"  [DRIFT] w{i+1} pos={pos:,}  MMD={mmd:.4f}  buf={len(warn_buffer_X)}")
            tsne_coords = _tsne_2d(win_np)
            X_sel, y_sel, _ = select_drift_samples_per_class(
                ref_np, win_np, X_win, y_win, tsne_coords
            )
            # combine hull-selected samples with warning buffer
            if warn_buffer_X:
                X_new = np.concatenate([np.array(warn_buffer_X), X_sel])
                y_new = np.concatenate([np.array(warn_buffer_y), y_sel])
            else:
                X_new, y_new = X_sel, y_sel

            # experience replay: mix in 50% burn-in to prevent catastrophic forgetting
            n_replay   = len(X_new) // 2
            replay_idx = np.random.choice(len(burnin_texts), n_replay, replace=False)
            X_train    = np.concatenate([X_new, burnin_texts[replay_idx]])
            y_train    = np.concatenate([y_new, burnin_labels[replay_idx]])
            perm       = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]

            w_before = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            train_model(model, tokenizer, X_train, y_train, label_map,
                        label="adapt", epochs=ADAPT_EPOCHS, lr=5e-5)
            changed = sum(1 for n, p in model.named_parameters()
                          if p.requires_grad and not torch.equal(w_before[n], p.data))
            print(f"  [LORA CHECK] {changed}/{len(w_before)} params changed | "
                  f"{len(X_new)} drift + {n_replay} replay = {len(X_train)} total")

            new_ref_t, _ = encode_and_predict(X_win, model, tokenizer)
            ref_np = new_ref_t.cpu().numpy()
            gamma  = _estimate_gamma(ref_np)
            det.reset()
            warn_buffer_X = []
            warn_buffer_y = []
            adapt_positions.append(pos)

        positions.append(pos)
        accuracies.append(acc)

    return positions, accuracies, adapt_positions


# ── Main ───────────────────────────────────────────────────────────────────────

_init_summary_csv()

total_runs = len(DATASETS) * len(ENCODERS)
print(f"\nExperiment 16 — Baseline vs Retrain: {total_runs} comparisons")
print(f"  Retrain config: {SELECTION_METHOD}, det={DETECTOR_NAME}, "
      f"metric={METRIC_NAME}, W={WINDOW_SIZE}\n")

completed = 0

for dataset_file in DATASETS:
    com_path = f"experiments/{dataset_file}"
    dataset_name = dataset_file.replace(".csv", "")
    print(f"\n{'#'*70}")
    print(f"  Dataset: {com_path}")
    print(f"{'#'*70}")

    com_df = pd.read_csv(com_path).dropna(subset=[TEXT_COL])
    if SORT_BY_DATE and "created_at" in com_df.columns:
        com_df = com_df.sort_values("created_at").reset_index(drop=True)
    all_texts  = com_df[TEXT_COL].astype(str).values[:N_DRIFTED]
    all_labels = com_df["label"].values[:N_DRIFTED]

    burnin_texts  = all_texts[:BURNIN_SIZE]
    burnin_labels = all_labels[:BURNIN_SIZE]
    stream_texts  = all_texts[BURNIN_SIZE:]
    stream_labels = all_labels[BURNIN_SIZE:]

    label_map     = {v: i for i, v in enumerate(np.unique(all_labels))}
    inv_label_map = {i: v for v, i in label_map.items()}
    num_labels    = len(label_map)

    for enc in ENCODERS:
        run_tag = f"{dataset_name}_{enc['name'].lower()}_cmp"

        plot_path = os.path.join(PLOTS_DIR, f"{run_tag}.png")
        if os.path.exists(plot_path):
            print(f"  [SKIP]  {run_tag} (already done)")
            completed += 1
            continue

        print(f"\n{'='*60}")
        print(f"  Run {completed+1}/{total_runs}: {run_tag}")
        print(f"{'='*60}")

        t_start = time.perf_counter()

        # build and train on burnin once
        model, tokenizer = build_model(enc, num_labels)
        model.print_trainable_parameters()
        print(f"  [LORA TARGETS] {_detect_lora_targets(model.base_model.model)}")
        train_model(model, tokenizer, burnin_texts, burnin_labels,
                    label_map, label="burnin")
        initial_state = copy.deepcopy(model.state_dict())

        # ── baseline pass ──────────────────────────────────────────────────
        print(f"\n  [BASELINE] pass...")
        model.load_state_dict(copy.deepcopy(initial_state))
        pos_base, acc_base = pass_baseline(
            model, tokenizer, inv_label_map, stream_texts, stream_labels
        )

        # ── retrain pass ───────────────────────────────────────────────────
        print(f"\n  [RETRAIN] pass...")
        model.load_state_dict(copy.deepcopy(initial_state))
        pos_ret, acc_ret, adapt_pos = pass_retrain(
            model, tokenizer, label_map, inv_label_map,
            stream_texts, stream_labels
        )

        wall_time = time.perf_counter() - t_start

        save_comparison_plot(
            pos_base, acc_base,
            pos_ret,  acc_ret,
            adapt_pos,
            run_tag, dataset_name, "", enc["name"],
        )

        post_drift_base = [acc_base[j] for j, p in enumerate(pos_base)
                           if p >= DRIFT_POSITIONS[0]]
        post_drift_ret  = [acc_ret[j]  for j, p in enumerate(pos_ret)
                           if p >= DRIFT_POSITIONS[0]]

        _append_summary({
            "run_tag":                    run_tag,
            "dataset":                    dataset_name,
            "base":                       dataset_name,
            "subset":                     "",
            "encoder":                    enc["name"],
            "mean_acc_baseline":          round(float(np.mean(acc_base)), 4),
            "mean_acc_retrain":           round(float(np.mean(acc_ret)),  4),
            "mean_acc_post_drift_baseline": round(float(np.mean(post_drift_base)), 4) if post_drift_base else "n/a",
            "mean_acc_post_drift_retrain":  round(float(np.mean(post_drift_ret)),  4) if post_drift_ret  else "n/a",
            "n_adaptations":              len(adapt_pos),
            "wall_time_s":                round(wall_time, 1),
        })

        completed += 1
        print(f"  [DONE]  {run_tag}  adaptations={len(adapt_pos)}  "
              f"wall_time={round(wall_time,1)}s  ({completed}/{total_runs})")

        del model, tokenizer, initial_state
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

print(f"\n{'='*70}")
print(f"Experiment 16 complete.  {completed}/{total_runs} runs.")
print(f"  Summary → {SUMMARY_CSV}")
print(f"  Plots   → {PLOTS_DIR}/")
