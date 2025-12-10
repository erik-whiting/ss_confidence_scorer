import os
import json
import time
import joblib
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score


class RNADataset(Dataset):
    # So we can use DataLoader
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Configs
EASY_CSV_TRAINING = "./scripts/training_data_easy.csv"
EASY_CSV_VALIDATION = "./scripts/easy_validation.csv"
EASY_CSV_TESTING = "./scripts/easy_testing.csv"

HARD_CSV_TRAINING = "./scripts/training_data_hard.csv"
HARD_CSV_VALIDATION = "./scripts/hard_validation.csv"
HARD_CSV_TESTING = "./scripts/hard_testing.csv"

FEATURES = [
    "sequence_length","gc_content","sequence_entropy","mfe","ens_def",
    "longest_sequential_A","longest_sequential_C","longest_sequential_U","longest_sequential_G",
    "longest_GC_helix","GU_pairs","rate_of_bps_predicted",
    "hairpin_count","junction_count","helix_count","singlestrand_count",
    "mway_junction_count","AU_pairs_in_helix_terminal_ends",
    "helices_with_reverse_complement","hairpins_with_gt4_unpaired_nts",
]

BUNDLE_PATH = "quality_model_bundle2.joblib"

# --- training
easy_training = pd.read_csv(EASY_CSV_TRAINING).assign(target=1)
hard_training = pd.read_csv(HARD_CSV_TRAINING).assign(target=0)
training_df = pd.concat([easy_training, hard_training], ignore_index=True)
X_training = training_df[FEATURES].astype(float).to_numpy(dtype=np.float32)
y_training = training_df["target"].to_numpy(dtype=np.float32)

# --- validation
easy_validation = pd.read_csv(EASY_CSV_VALIDATION).assign(target=1)
hard_validation = pd.read_csv(HARD_CSV_VALIDATION).assign(target=0)
validation_df = pd.concat([easy_validation, hard_validation], ignore_index=True)
X_validation = validation_df[FEATURES].astype(float).to_numpy(dtype=np.float32)
y_validation = validation_df["target"].to_numpy(dtype=np.float32)

# --- testing
easy_testing = pd.read_csv(EASY_CSV_TESTING).assign(target=1)
hard_testing = pd.read_csv(HARD_CSV_TESTING).assign(target=0)
testing_df = pd.concat([easy_testing, hard_testing], ignore_index=True)
X_testing = testing_df[FEATURES].astype(float).to_numpy(dtype=np.float32)
y_testing = testing_df["target"].to_numpy(dtype=np.float32)

# ensure no leakage:
assert set(training_df["datapoint"]).isdisjoint(validation_df["datapoint"])
assert set(training_df["datapoint"]).isdisjoint(testing_df["datapoint"])
assert set(validation_df["datapoint"]).isdisjoint(testing_df["datapoint"])

# Scale training data
scaler = StandardScaler().fit(X_training)
Xtr = scaler.transform(X_training).astype(np.float32)
Xva = scaler.transform(X_validation).astype(np.float32)
Xte = scaler.transform(X_testing).astype(np.float32)
ytr, yva, yte = y_training, y_validation, y_testing # shorthand for the variable names

# Model-specific configs
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15        # remaining is train
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = (64, 32)

train_loader = DataLoader(RNADataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
val_loader =   DataLoader(RNADataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False)
test_loader =  DataLoader(RNADataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False)

# Model
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, HIDDEN[0]), nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(),
            nn.Linear(HIDDEN[1], 1),
        )
    def forward(self, x): return self.net(x).squeeze(1)  # logits


model = MLP(Xtr.shape[1]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.BCEWithLogitsLoss()
best_state = None
best_val_auc = -float("inf")

# To evaluate the probabilities
def eval_probs(loader):
    model.eval()
    outs=[]
    ys=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(DEVICE)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            outs.append(p)
            ys.append(yb.numpy())

    return np.concatenate(outs).ravel(), np.concatenate(ys).ravel()



for epoch in range(1, EPOCHS+1):
    model.train()
    runloss=0.0

    for xb,yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        runloss += loss.item()*len(xb)

    # validate
    pv, yv = eval_probs(val_loader)
    val_auc = roc_auc_score(yv, pv) if len(np.unique(yv))>1 else float("nan")

    if np.isfinite(val_auc) and val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

    print(f"Epoch {epoch:03d}  loss={runloss/len(Xtr):.4f}  val_auc={val_auc:.4f}")

if best_state is not None:
    model.load_state_dict(best_state)


# Picking the low/high cutoffs
def choose_ternary_thresholds(y_true, y_prob, target_precision=0.9):
    # low: prob < t_low --> "bad"
    t_low = 0.2
    for t in np.linspace(0.05, 0.5, 20):
        pred_bad = (y_prob < t)
        tp = np.sum(pred_bad & (y_true==0))
        fp = np.sum(pred_bad & (y_true==1))
        prec_bad = tp/(tp+fp) if (tp+fp)>0 else 0.0
        if prec_bad >= target_precision:
            t_low = float(t)
            break

    # high: prob >= t_high --> "good"
    t_high = 0.8
    for t in np.linspace(0.5, 0.95, 20):
        pred_good = (y_prob >= t)
        tp = np.sum(pred_good & (y_true==1))
        fp = np.sum(pred_good & (y_true==0))
        prec_good = tp/(tp+fp) if (tp+fp)>0 else 0.0
        if prec_good >= target_precision:
            t_high = float(t)
            break
    return t_low, t_high

print(f"\n\nNow running validations to pick thresholds")
pv, yv = eval_probs(val_loader)
t_low, t_high = choose_ternary_thresholds(yv, pv, target_precision=0.9)
print("Chosen thresholds:", t_low, t_high)

# more testing
print(f"\n\nRunning tests with saved thresholds {t_low}, and {t_high}")
pt, yt = eval_probs(test_loader)
yhat = (pt >= 0.5).astype(int)
acc = accuracy_score(yt, yhat)
p,r,f1,_ = precision_recall_fscore_support(yt, yhat, average="binary", zero_division=0)
auc = roc_auc_score(yt, pt) if len(np.unique(yt))>1 else float("nan")
print({"test_auc": auc, "test_acc": acc, "test_precision": p, "test_recall": r, "test_f1": f1})

# Save
bundle = {
    "meta": {"features": FEATURES, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
    "scaler": scaler,
    "model_state_dict": {k: v.cpu() for k,v in model.state_dict().items()},
    "model_class": "MLP",
    "model_kwargs": {"d": Xtr.shape[1]},
    "thresholds": {"low": t_low, "high": t_high},
    "metrics": {"val_auc": float(best_val_auc), "test_auc": float(auc)},
}
joblib.dump(bundle, BUNDLE_PATH)
print(f"Saved to {BUNDLE_PATH}")

