"""
Baseline experiment:
  datasets × encoders  — NO LoRA retraining, no drift detector.

  One pass per (dataset, encoder): encode stream window by window,
  track prequential accuracy, MMD and JSD against the burn-in reference,
  and save a 3-panel plot + per-window CSV + summary row.

Grid:
  datasets : DATASETS list
  encoders : ModernBERT, Jina-v2

Fixed:
  window_size : 100
  n_ref_emb   : 500 (full burn-in)

Outputs (per run):
  results/per_window/{tag}.csv   ← per-window detail
  results/plots/{tag}.png        ← accuracy / MMD / JSD timeline
  results/summary.csv            ← one row per run (appended)
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import csv
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist

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
RESULTS_DIR    = "baseline_results"
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

WINDOW_SIZE         = 16
ACC_PLOT_WINDOW     = WINDOW_SIZE * 4   # rolling window for prequential accuracy

BURNIN_EPOCHS = 3
TRAIN_BATCH   = 16

# ── Dataset list ───────────────────────────────────────────────────────────────
DATASETS = [
    ("airbnb", 1), ("airbnb", 2), ("airbnb", 3), ("airbnb", 4), ("airbnb", 5),
]

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
    "run_tag",
    "dataset", "base", "subset", "encoder",
    "window_size",
    "wall_time_s",
    "n_windows",
    "mean_acc", "mean_acc_post_drift",
    "mean_mmd", "mean_jsd",
    "ref_acc_initial", "ref_entropy_initial",
]

DETAIL_FIELDS = [
    "window_idx", "window_pos",
    "prequential_acc", "entropy",
    "mmd", "jsd",
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


# ── Plot ───────────────────────────────────────────────────────────────────────

def save_results_plot(
    window_positions, window_accuracies,
    mmd_scores, jsd_scores,
    ref_acc, run_tag, base, subset, encoder_name,
):
    drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{base}-comdrift-{subset}  |  enc={encoder_name}  "
        f"[BASELINE — no retraining]",
        fontsize=12,
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
          f"Prequential acc (rolling {ACC_PLOT_WINDOW})",
          ref=ref_acc, ref_label=f"Ref acc ({ref_acc:.3f})", ylabel="Accuracy")
    axes[0].set_ylim(0, 1.05)

    _fill(axes[1], mmd_scores, "darkorange", "MMD", ylabel="MMD")
    _fill(axes[2], jsd_scores, "teal",       "JSD",
          ref=np.log(2), ref_label=f"Max JSD ({np.log(2):.3f})", ylabel="JSD (nats)")

    axes[2].set_xlabel("Position in stream", fontsize=10)

    legend_patches = []
    for pos, col in zip(DRIFT_POSITIONS, drift_colors):
        for ax in axes:
            ax.axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)
        legend_patches.append(mpatches.Patch(color=col, label=f"Drift @ {pos//1_000}k"))

    x_ticks = np.arange(0, N_DRIFTED + 1, 25_000)
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels([f"{x//1_000}k" for x in x_ticks])
    axes[2].set_xlim(0, N_DRIFTED)
    axes[2].legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"{run_tag}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT]  → {fname}")


# ── Core run ───────────────────────────────────────────────────────────────────

def run_baseline(
    model, tokenizer, label_map, inv_label_map,
    burnin_texts, burnin_labels,
    stream_texts, stream_labels,
    base: str, subset: int, encoder_name: str,
    run_tag: str,
) -> dict:
    """One baseline pass — no drift detection, no retraining."""

    n_windows = len(stream_texts) // WINDOW_SIZE

    # encode burn-in reference once
    ref_embs_t, ref_preds, ref_ent_arr = encode_and_predict(
        burnin_texts, model, tokenizer
    )
    ref_np = ref_embs_t.cpu().numpy()

    ref_preds_orig  = np.array([inv_label_map[p] for p in ref_preds])
    ref_acc_initial = float((ref_preds_orig == burnin_labels).mean())
    ref_entropy_ini = float(ref_ent_arr.mean())

    gamma = _estimate_gamma(ref_np)

    correct_history: list[int] = []
    window_positions  = []
    window_accuracies = []
    mmd_scores        = []
    jsd_scores        = []
    per_window_rows   = []

    t_start = time.perf_counter()

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end   = start + WINDOW_SIZE
        X_win = stream_texts[start:end]
        y_win = stream_labels[start:end]

        win_embs_t, win_preds, win_ent = encode_and_predict(X_win, model, tokenizer)
        win_np = win_embs_t.cpu().numpy()

        preds_orig = np.array([inv_label_map[p] for p in win_preds])
        correct_history.extend((preds_orig == y_win).astype(int).tolist())
        acc     = float(np.mean(correct_history[-ACC_PLOT_WINDOW:]))
        entropy = float(win_ent.mean())
        mmd     = compute_mmd(ref_np, win_np, gamma)
        jsd     = compute_js_divergence(ref_np, win_np)
        pos     = BURNIN_SIZE + start + WINDOW_SIZE // 2

        window_positions.append(pos)
        window_accuracies.append(acc)
        mmd_scores.append(mmd)
        jsd_scores.append(jsd)

        per_window_rows.append({
            "window_idx": i + 1, "window_pos": pos,
            "prequential_acc": round(acc, 4),
            "entropy": round(entropy, 4),
            "mmd": round(mmd, 6),
            "jsd": round(jsd, 6),
        })

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{run_tag}] {i+1}/{n_windows}  "
                  f"acc={acc:.3f}  mmd={mmd:.4f}  jsd={jsd:.4f}  "
                  f"elapsed={elapsed:.0f}s")

    wall_time = time.perf_counter() - t_start

    # per-window CSV
    pw_path = os.path.join(PER_WINDOW_DIR, f"{run_tag}.csv")
    _save_per_window_csv(pw_path, per_window_rows)
    print(f"  [CSV]   → {pw_path}")

    # plot
    save_results_plot(
        window_positions, window_accuracies,
        mmd_scores, jsd_scores,
        ref_acc_initial, run_tag, base, subset, encoder_name,
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
        "window_size":         WINDOW_SIZE,
        "wall_time_s":         round(wall_time, 1),
        "n_windows":           n_windows,
        "mean_acc":            round(float(np.mean(window_accuracies)), 4),
        "mean_acc_post_drift": round(float(np.mean(post_drift_accs)), 4) if post_drift_accs else "n/a",
        "mean_mmd":            round(float(np.mean(mmd_scores)), 6),
        "mean_jsd":            round(float(np.mean(jsd_scores)), 6),
        "ref_acc_initial":     round(ref_acc_initial, 4),
        "ref_entropy_initial": round(ref_entropy_ini, 4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

_init_summary_csv()

total_runs = len(DATASETS) * len(ENCODERS)
print(f"\nBaseline experiment: {total_runs} total runs")
print(f"  datasets={len(DATASETS)}  encoders={len(ENCODERS)}  window_size={WINDOW_SIZE}\n")

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
        run_tag = f"{base}{subset}_{enc['name'].lower()}_baseline"

        pw_path = os.path.join(PER_WINDOW_DIR, f"{run_tag}.csv")
        if os.path.exists(pw_path):
            print(f"  [SKIP]  {run_tag} (already done)")
            completed += 1
            continue

        print(f"\n{'='*60}")
        print(f"  Run {completed+1}/{total_runs}: {run_tag}")
        print(f"{'='*60}")

        model, tokenizer = build_model(enc, num_labels)
        model.print_trainable_parameters()
        train_model(model, tokenizer, burnin_texts, burnin_labels,
                    label_map, label="burnin")

        summary = run_baseline(
            model=model,
            tokenizer=tokenizer,
            label_map=label_map,
            inv_label_map=inv_label_map,
            burnin_texts=burnin_texts,
            burnin_labels=burnin_labels,
            stream_texts=stream_texts,
            stream_labels=stream_labels,
            base=base,
            subset=subset,
            encoder_name=enc["name"],
            run_tag=run_tag,
        )

        _append_summary(summary)
        completed += 1
        print(f"  [DONE]  {run_tag}  wall_time={summary['wall_time_s']}s  "
              f"({completed}/{total_runs} runs)")

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

print(f"\n{'='*70}")
print(f"Baseline complete.  {completed}/{total_runs} runs.")
print(f"  Summary    → {SUMMARY_CSV}")
print(f"  Per-window → {PER_WINDOW_DIR}/")
print(f"  Plots      → {PLOTS_DIR}/")
