# train_quality_model.py
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


# ---------------------------
# Config
# ---------------------------
SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = (64, 32)           # MLP sizes
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15        # remaining is train
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EASY_CSV = "chem_map_all_easy_preds_enriched.csv"
HARD_CSV = "chem_map_all_hard_preds_enriched.csv"

FEATURES = [
    "sequence_length","gc_content","sequence_entropy","mfe","ens_def",
    "longest_sequential_A","longest_sequential_C","longest_sequential_U","longest_sequential_G",
    "longest_GC_helix","GU_pairs","rate_of_bps_predicted",
    "hairpin_count","junction_count","helix_count","singlestrand_count",
    "mway_junction_count","AU_pairs_in_helix_terminal_ends",
    "helices_with_reverse_complement","hairpins_with_gt4_unpaired_nts",
]

BUNDLE_PATH = "quality_model_bundle.joblib"  # everything saved here


# ---------------------------
# Repro
# ---------------------------
def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------
# Data
# ---------------------------
def load_and_prepare() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    easy = pd.read_csv(EASY_CSV)
    hard = pd.read_csv(HARD_CSV)
    easy = easy.assign(target=1)
    hard = hard.assign(target=0)

    df = pd.concat([easy, hard], ignore_index=True)

    # Keep only rows with all required features present
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    X = df[FEATURES].astype(float).to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.float32)
    groups = df["datapoint"].astype(str).to_numpy()

    return df, X, y, groups


class RNADataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN[0]), nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(),
            nn.Linear(HIDDEN[1], 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)  # logits


# ---------------------------
# Split by datapoint groups
# ---------------------------
def grouped_splits(groups: np.ndarray, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    all_idx = np.arange(len(groups))
    train_val_idx, test_idx = next(gss.split(all_idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac/(1-test_frac), random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val_idx, groups=groups[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]
    return train_idx, val_idx, test_idx


# ---------------------------
# Metrics / thresholds
# ---------------------------
def eval_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"auc": auc, "acc": acc, "precision": p, "recall": r, "f1": f1}

def choose_ternary_thresholds(y_true, y_prob, target_precision=0.9):
    """
    Pick low/high cutoffs: below low -> "bad", above high -> "good", otherwise "uncertain".
    We pick the smallest threshold achieving the target precision for each class on val.
    """
    order = np.argsort(y_prob)
    probs_sorted = y_prob[order]
    y_sorted = y_true[order]

    # Low threshold for "bad": look at left tail (predict bad if prob < t_low)
    best_low = 0.2  # fallback
    for t in np.linspace(0.05, 0.5, 20):
        pred_bad = (y_prob < t).astype(int)  # 1 for bad prediction claim
        # precision for "bad" = TP_bad / (TP_bad + FP_bad) where true bad is y=0
        tp = ((pred_bad==1) & (y_true==0)).sum()
        fp = ((pred_bad==1) & (y_true==1)).sum()
        prec_bad = tp / (tp+fp) if (tp+fp)>0 else 0.0
        if prec_bad >= target_precision:
            best_low = t
            break

    # High threshold for "good": look at right tail (predict good if prob >= t_high)
    best_high = 0.8  # fallback
    for t in np.linspace(0.5, 0.95, 20):
        pred_good = (y_prob >= t).astype(int)  # 1 for good claim
        tp = ((pred_good==1) & (y_true==1)).sum()
        fp = ((pred_good==1) & (y_true==0)).sum()
        prec_good = tp / (tp+fp) if (tp+fp)>0 else 0.0
        if prec_good >= target_precision:
            best_high = t
            break

    return float(best_low), float(best_high)


# ---------------------------
# Training loop
# ---------------------------
def train():
    set_seed()
    df, X, y, groups = load_and_prepare()

    # Grouped splits
    tr_idx, va_idx, te_idx = grouped_splits(groups)

    # Preprocessing on TRAIN only
    scaler = StandardScaler().fit(X[tr_idx])
    X_tr = scaler.transform(X[tr_idx]).astype(np.float32)
    X_va = scaler.transform(X[va_idx]).astype(np.float32)
    X_te = scaler.transform(X[te_idx]).astype(np.float32)

    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # DataLoaders
    train_loader = DataLoader(RNADataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(RNADataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(RNADataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = MLP(in_dim=X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -np.inf
    best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)

        # Validate
        model.eval()
        with torch.no_grad():
            yv = []; pv = []
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                pv.append(prob); yv.append(yb.numpy())
            pv = np.concatenate(pv); yv = np.concatenate(yv)

        val_auc = roc_auc_score(yv, pv) if len(np.unique(yv))>1 else float("nan")

        if np.isfinite(val_auc) and val_auc > best_val:
            best_val = val_auc
            best_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_auc": best_val
            }

        print(f"Epoch {epoch:03d} | train_loss={running/len(train_loader.dataset):.4f} | val_auc={val_auc:.4f}")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])

    # Final validation metrics and thresholds
    model.eval()
    with torch.no_grad():
        pv = []
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            pv.append(torch.sigmoid(model(xb)).cpu().numpy())
        pv = np.concatenate(pv)

    t_low, t_high = choose_ternary_thresholds(y_va, pv, target_precision=0.9)

    # Test metrics
    with torch.no_grad():
        pt = []
        yt = []
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            pt.append(torch.sigmoid(model(xb)).cpu().numpy()); yt.append(yb.numpy())
        pt = np.concatenate(pt); yt = np.concatenate(yt)

    metrics = {
        "val": eval_metrics(y_va, pv, threshold=0.5),
        "test": eval_metrics(yt, pt, threshold=0.5),
        "val_thresholds": {"low": t_low, "high": t_high, "target_precision": 0.9},
        "split_sizes": {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)},
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    # Save bundle
    bundle = {
        "meta": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "features": FEATURES,
            "device_trained": DEVICE,
            "seed": SEED,
        },
        "scaler": scaler,
        "model_state_dict": model.state_dict(),
        "model_class": "MLP",
        "model_kwargs": {"in_dim": X.shape[1]},
        "thresholds": {"low": t_low, "high": t_high},
        "metrics": metrics,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"Saved bundle to {BUNDLE_PATH}")


# ---------------------------
# Inference utilities
# ---------------------------
def load_bundle(path=BUNDLE_PATH):
    b = joblib.load(path)
    model = MLP(**b["model_kwargs"])
    model.load_state_dict(b["model_state_dict"])
    model.eval()
    return b, model

def predict_quality(attrs: Dict[str, float], bundle_path=BUNDLE_PATH) -> Dict[str, object]:
    """
    attrs: dict mapping each feature name to its numeric value.
    Returns: prob_good (float), decision (str)
    """
    b, model = load_bundle(bundle_path)
    feats = b["meta"]["features"]
    scaler: StandardScaler = b["scaler"]
    t_low = b["thresholds"]["low"]; t_high = b["thresholds"]["high"]

    # align & impute with training means if missing

    # row = pd.Series(attrs, dtype=float).reindex(feats)
    # if row.isna().any():
    #     means = pd.Series(scaler.mean_, index=feats)
    #     row = row.fillna(means)

    # X_new = scaler.transform(pd.DataFrame([row.values], columns=feats)).astype(np.float32)

    row = pd.Series(attrs, dtype=float).reindex(feats)
    if row.isna().any():
        means = pd.Series(scaler.mean_, index=feats)
        row = row.fillna(means)

    if hasattr(scaler, "feature_names_in_"):  # scaler fit on a DataFrame (with names)
        # Optional safety check: ensure same columns/order
        assert list(feats) == list(scaler.feature_names_in_)
        X_new = scaler.transform(pd.DataFrame([row.values], columns=feats)).astype(np.float32)
    else:  # scaler fit on a NumPy array (no names)
        X_new = scaler.transform(row.to_numpy(dtype=float).reshape(1, -1)).astype(np.float32)


    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X_new))).numpy().ravel()[0].item()

    if prob >= t_high:
        decision = "probably a good prediction"
    elif prob < t_low:
        decision = "probably a bad prediction"
    else:
        decision = "cannot determine"

    return {"prob_good": float(prob), "decision": decision, "thresholds": {"low": t_low, "high": t_high}}

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    train()
