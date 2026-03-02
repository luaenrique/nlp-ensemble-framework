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
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from peft import get_peft_model, LoraConfig, TaskType

import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)

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

clf = Pipeline([
    ("vec", HashingVectorizer(
        n_features=2 ** 16,
        ngram_range=(1, 2),
        alternate_sign=False,
    )),
    ("cls", SGDClassifier(
        loss="log_loss",
        max_iter=5,
        random_state=42,
        n_jobs=-1,
    )),
])

print("\nTraining on reference window …")
clf.fit(ref_texts, ref_labels)
ref_acc = (clf.predict(ref_texts) == ref_labels).mean()
print(f"Reference train accuracy: {ref_acc:.3f}")

n_windows         = len(stream_texts) // WINDOW_SIZE
window_positions  = [] 
window_accuracies = []
window_entropies  = []  

classes = np.unique(ref_labels)


device = "mps" if torch.backends.mps.is_available() else "cuda"

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["Wqkv"],
    task_type=TaskType.FEATURE_EXTRACTION,
)

model = get_peft_model(model, lora_config).to(device)
model.print_trainable_parameters()



def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)


def encode(texts, batch_size = 32):
    all_embeddings = []
    model.eval()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        cls_token_embedding = outputs.hidden_states[-1][:,0,:].cpu().numpy()
        
        all_embeddings.append(cls_token_embedding)
    
    return np.concatenate(all_embeddings, axis=0)



N_REF_EMB = 400
ref_embs_t = torch.tensor(encode(ref_texts[:N_REF_EMB]), dtype=torch.float32).to(device)

window_positions = []
mmd_scores       = []        

for i in range(n_windows):
    start_batch_pos = i * WINDOW_SIZE
    end_batch_pos   = start_batch_pos + WINDOW_SIZE
    

    X_win = stream_texts[start_batch_pos:end_batch_pos]
    y_win = stream_labels[start_batch_pos:end_batch_pos]
    
    win_embs_t = torch.tensor(encode(X_win), dtype=torch.float32).to(device)
    score = MMD(ref_embs_t, win_embs_t, kernel="rbf").item()

    window_positions.append(start_batch_pos + WINDOW_SIZE // 2)
    mmd_scores.append(score)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(window_positions, mmd_scores, color="steelblue", linewidth=1.3)
ax.set_ylabel("MMD score")
ax.set_xlabel("Position in stream")
ax.grid(alpha=0.3)

drift_colors = ["#e74c3c", "#e67e22", "#9b59b6"]
for pos, col in zip(DRIFT_POSITIONS, drift_colors):
    ax.axvline(x=pos, color=col, linestyle="--", linewidth=1.8, alpha=0.85)

plt.tight_layout()
plt.savefig("mmd_drift_detection.png", dpi=150, bbox_inches="tight")
plt.show()