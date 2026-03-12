"""
Experiment 09: ModernBERT — LoRA retraining vs. no retraining comparison.

Two conditions run sequentially on the same data/model init:
  A) WITH LoRA adapt on confirmed drift
  B) WITHOUT any adaptation

Reports per-window accuracy and mean accuracy for each condition.
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from river.drift import ADWIN

from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertForSequenceClassification

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "experiments/tech_non_tech_dataset.csv"
MODEL_NAME   = "answerdotai/ModernBERT-base"

BURNIN_SIZE  = 1000
WINDOW_SIZE  = 200
N_REF_EMB    = 200
TRAIN_EPOCHS = 3
TRAIN_BATCH  = 16
MMD_THRESHOLD = 0.2

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}")

# ── Load data ──────────────────────────────────────────────────────────────────
com_df = (
    pd.read_csv(DATASET_PATH)
    .dropna(subset=["text"])
    .sort_values("created_at")
    .reset_index(drop=True)
)
all_texts  = com_df["text"].astype(str).values
all_labels = com_df["label"].values

label_map     = {v: i for i, v in enumerate(np.unique(all_labels))}
inv_label_map = {i: v for v, i in label_map.items()}
num_labels    = len(label_map)

burnin_texts  = all_texts[:BURNIN_SIZE]
burnin_labels = all_labels[:BURNIN_SIZE]
stream_texts  = all_texts[BURNIN_SIZE:]
stream_labels = all_labels[BURNIN_SIZE:]
n_windows     = len(stream_texts) // WINDOW_SIZE

print(f"Burn-in: {len(burnin_texts):,}  |  Stream: {len(stream_texts):,}  |  Windows: {n_windows}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _estimate_gamma(embeddings: np.ndarray) -> float:
    if len(embeddings) > 200:
        idx = np.random.choice(len(embeddings), 200, replace=False)
        embeddings = embeddings[idx]
    sq_dists = pdist(embeddings, metric="sqeuclidean")
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

    K_XX = rbf(X, X); np.fill_diagonal(K_XX, 0.0)
    K_YY = rbf(Y, Y); np.fill_diagonal(K_YY, 0.0)
    K_XY = rbf(X, Y)

    t_XX = K_XX.sum() / (n * (n - 1)) if n > 1 else 0.0
    t_YY = K_YY.sum() / (m * (m - 1)) if m > 1 else 0.0
    return float(max(0.0, t_XX + t_YY - 2.0 * K_XY.mean()))


def _detect_lora_targets(model) -> list[str]:
    candidates = {"query", "key", "value", "q_proj", "k_proj", "v_proj", "Wqkv"}
    found = {name.split(".")[-1] for name, mod in model.named_modules()
             if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in candidates}
    return list(found) if found else ["query", "value"]


def encode_and_predict(texts, model, tokenizer, batch_size=32):
    all_embs, all_preds = [], []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        all_embs.append(out.hidden_states[-1][:, 0, :].cpu())
        all_preds.append(out.logits.argmax(dim=-1).cpu().numpy())
    return torch.cat(all_embs, dim=0).numpy(), np.concatenate(all_preds)


def train_model(model, tokenizer, texts, labels, label="train"):
    # Only optimize trainable (LoRA) params — base model is frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=2e-4)
    model.train()
    for epoch in range(TRAIN_EPOCHS):
        epoch_loss = 0.0
        for i in range(0, len(texts), TRAIN_BATCH):
            bt = list(texts[i : i + TRAIN_BATCH])
            bl = torch.tensor([label_map[l] for l in labels[i : i + TRAIN_BATCH]]).to(device)
            inp = tokenizer(bt, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            loss = model(**inp, labels=bl).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print(f"    [{label}] Epoch {epoch+1}/{TRAIN_EPOCHS}  loss={epoch_loss:.4f}")


# ── Build and train base model (shared starting point) ────────────────────────
print("\nLoading ModernBERT …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = ModernBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)

lora_targets = _detect_lora_targets(base_model)
print(f"LoRA targets: {lora_targets}")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=lora_targets,
                         task_type=TaskType.SEQ_CLS)
base_model = get_peft_model(base_model, lora_config).to(device)
base_model.print_trainable_parameters()

print("\nTraining on burn-in window …")
train_model(base_model, tokenizer, burnin_texts, burnin_labels, label="burnin")
print("Burn-in training done.\n")

# Save state after burn-in — deepcopy of PEFT models on MPS/CUDA is unreliable;
# state_dict approach guarantees a clean copy on the correct device.
burnin_state = {k: v.clone() for k, v in base_model.state_dict().items()}


def _fresh_model() -> torch.nn.Module:
    """Create a new PEFT model loaded with the post-burn-in weights."""
    m = ModernBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16,
                                     target_modules=lora_targets,
                                     task_type=TaskType.SEQ_CLS))
    m.load_state_dict(burnin_state)
    return m.to(device)


# ── Run one condition ──────────────────────────────────────────────────────────

def run_condition(model, do_retrain: bool, label: str):
    """Run the stream evaluation loop with or without LoRA retraining on drift."""
    model = _fresh_model()   # isolated copy via state_dict — safe on MPS/CUDA

    ref_np, ref_preds = encode_and_predict(burnin_texts[:N_REF_EMB], model, tokenizer)
    ref_acc = (np.array([inv_label_map[p] for p in ref_preds]) == burnin_labels[:N_REF_EMB]).mean()
    gamma   = _estimate_gamma(ref_np)

    detector     = ADWIN(delta=0.001)
    accuracies   = []
    adapt_events = []

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        X_win = stream_texts[start : start + WINDOW_SIZE]
        y_win = stream_labels[start : start + WINDOW_SIZE]

        win_np, win_preds = encode_and_predict(X_win, model, tokenizer)
        preds_orig = np.array([inv_label_map[p] for p in win_preds])
        acc        = (preds_orig == y_win).mean()
        mmd        = compute_mmd(ref_np, win_np, gamma)

        detector.update(mmd)
        pos = BURNIN_SIZE + start + WINDOW_SIZE // 2

        if detector.drift_detected and do_retrain:
            print(f"  [{label}] Drift @ window {i+1} (pos {pos:,}) mmd={mmd:.4f} → retraining …")
            train_model(model, tokenizer, X_win, y_win, label="adapt")
            new_np, _ = encode_and_predict(X_win, model, tokenizer)
            ref_np  = new_np
            gamma   = _estimate_gamma(ref_np)
            detector = ADWIN(delta=0.001)
            adapt_events.append(i)
        elif detector.drift_detected:
            print(f"  [{label}] Drift @ window {i+1} (pos {pos:,}) mmd={mmd:.4f} (no retrain)")

        accuracies.append(acc)
        print(f"  [{label}] w{i+1:>3}/{n_windows}  acc={acc:.3f}  mmd={mmd:.4f}")

    mean_acc = float(np.mean(accuracies))
    print(f"\n  [{label}] Mean accuracy: {mean_acc:.4f}  (ref baseline: {ref_acc:.4f})")
    return accuracies, adapt_events, mean_acc, ref_acc


# ── Condition A: WITH retraining ───────────────────────────────────────────────
print("=" * 60)
print("Condition A: WITH LoRA retraining on drift")
print("=" * 60)
acc_retrain, adapt_A, mean_A, ref_acc = run_condition(base_model, do_retrain=True,  label="WITH-RETRAIN")

# ── Condition B: WITHOUT retraining ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Condition B: WITHOUT retraining")
print("=" * 60)
acc_no_retrain, adapt_B, mean_B, _ = run_condition(base_model, do_retrain=False, label="NO-RETRAIN")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Reference accuracy (burn-in):   {ref_acc:.4f}")
print(f"  Mean acc WITH retraining:        {mean_A:.4f}  (adapt events: {len(adapt_A)})")
print(f"  Mean acc WITHOUT retraining:     {mean_B:.4f}")
delta = mean_A - mean_B
print(f"  Delta (WITH - WITHOUT):          {delta:+.4f}  "
      f"({'retrain helps' if delta > 0 else 'retrain hurts or no effect'})")

# ── Plot ────────────────────────────────────────────────────────────────────────
window_positions = [BURNIN_SIZE + i * WINDOW_SIZE + WINDOW_SIZE // 2
                    for i in range(n_windows)]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(window_positions, acc_retrain,    label=f"WITH retraining  (mean={mean_A:.3f})",
        color="steelblue", linewidth=1.5)
ax.plot(window_positions, acc_no_retrain, label=f"WITHOUT retraining (mean={mean_B:.3f})",
        color="tomato",    linewidth=1.5, linestyle="--")
ax.axhline(ref_acc, color="gray", linestyle=":", linewidth=1,
           label=f"Reference acc ({ref_acc:.3f})")

for idx in adapt_A:
    ax.axvline(x=window_positions[idx], color="steelblue", linestyle=":", alpha=0.5)

ax.set_title("ModernBERT: LoRA retraining vs. no retraining\n"
             f"(burn-in={BURNIN_SIZE}, window={WINDOW_SIZE})", fontsize=12)
ax.set_xlabel("Stream position")
ax.set_ylabel("Window accuracy")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

out_path = "exp_09_retrain_vs_no_retrain.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved → {out_path}")
