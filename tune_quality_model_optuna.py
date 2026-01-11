import argparse
import time
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import optuna


# ---------- Dataset ----------
class RNADataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ---------- Model ----------
class MLP(nn.Module):
    def __init__(self, d: int, hidden=(64, 32), dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


@torch.no_grad()
def eval_probs(model, loader, device):
    model.eval()
    outs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        p = torch.sigmoid(model(xb)).cpu().numpy()
        outs.append(p)
        ys.append(yb.numpy())
    return np.concatenate(outs).ravel(), np.concatenate(ys).ravel()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(args):
    FEATURES = args.features

    # --- training
    easy_training = pd.read_csv(args.easy_train).assign(target=1)
    hard_training = pd.read_csv(args.hard_train).assign(target=0)
    train_df = pd.concat([easy_training, hard_training], ignore_index=True)

    # --- validation
    easy_val = pd.read_csv(args.easy_val).assign(target=1)
    hard_val = pd.read_csv(args.hard_val).assign(target=0)
    val_df = pd.concat([easy_val, hard_val], ignore_index=True)

    # --- testing (kept sacred; not used in objective)
    easy_test = pd.read_csv(args.easy_test).assign(target=1)
    hard_test = pd.read_csv(args.hard_test).assign(target=0)
    test_df = pd.concat([easy_test, hard_test], ignore_index=True)

    # leakage checks
    assert set(train_df["datapoint"]).isdisjoint(val_df["datapoint"])
    assert set(train_df["datapoint"]).isdisjoint(test_df["datapoint"])
    assert set(val_df["datapoint"]).isdisjoint(test_df["datapoint"])

    Xtr = train_df[FEATURES].astype(float).to_numpy(dtype=np.float32)
    ytr = train_df["target"].to_numpy(dtype=np.float32)

    Xva = val_df[FEATURES].astype(float).to_numpy(dtype=np.float32)
    yva = val_df["target"].to_numpy(dtype=np.float32)

    Xte = test_df[FEATURES].astype(float).to_numpy(dtype=np.float32)
    yte = test_df["target"].to_numpy(dtype=np.float32)

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    return (Xtr, ytr, Xva, yva, Xte, yte, scaler)


def objective(trial, args, Xtr, ytr, Xva, yva):
    device = args.device

    # --- search space
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    hidden = trial.suggest_categorical("hidden", [
        [16, 8],
        [32, 16],
        [64, 32],
        [128, 64],
        [256, 128],
        [128, 16],   # bottleneck
        [256, 32],   # stronger bottleneck
    ])
    hidden = tuple(hidden)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

    # early stop knobs per trial (optional)
    patience = trial.suggest_int("patience", 4, 10)
    min_delta = trial.suggest_float("min_delta", 1e-4, 1e-3, log=True)

    # fixed
    max_epochs = args.epochs
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(RNADataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RNADataset(Xva, yva), batch_size=512, shuffle=False)

    model = MLP(d=Xtr.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        pv, yv = eval_probs(model, val_loader, device)
        val_auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else float("nan")

        # report for pruning
        trial.report(val_auc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        improved = np.isfinite(val_auc) and (val_auc > best_val_auc + min_delta)
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # store best state in trial attributes (so we can rebuild later)
    trial.set_user_attr("best_val_auc", float(best_val_auc))
    trial.set_user_attr("hidden", hidden)
    trial.set_user_attr("dropout", float(dropout))
    trial.set_user_attr("batch_size", int(batch_size))
    # trial.set_user_attr("state_dict", best_state)  # ok for local runs; for big studies, save to disk instead

    return float(best_val_auc)


def choose_ternary_thresholds(
    y_true, y_prob,
    prec_bad=0.95,
    prec_good=0.95,
    min_gap=0.10,
    grid=np.linspace(0.01, 0.99, 99),
):
    best = None  # (coverage, t_low, t_high)

    for t_low in grid:
        for t_high in grid:
            if t_high < t_low + min_gap:
                continue

            bad = (y_prob < t_low)
            good = (y_prob >= t_high)

            tp_bad = np.sum(bad & (y_true == 0))
            fp_bad = np.sum(bad & (y_true == 1))
            p_bad = tp_bad / (tp_bad + fp_bad) if (tp_bad + fp_bad) else 0.0

            tp_good = np.sum(good & (y_true == 1))
            fp_good = np.sum(good & (y_true == 0))
            p_good = tp_good / (tp_good + fp_good) if (tp_good + fp_good) else 0.0

            if p_bad >= prec_bad and p_good >= prec_good:
                coverage = float((bad | good).mean())
                if best is None or coverage > best[0]:
                    best = (coverage, float(t_low), float(t_high))

    if best is None:
        return 0.4, 0.6
    _, t_low, t_high = best
    return t_low, t_high



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_quality.db")
    parser.add_argument("--study", type=str, default="rna_quality_mlp")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bundle_out", type=str, default="quality_model_bundle_optuna_best.joblib")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    # your CSVs
    parser.add_argument("--easy_train", type=str, required=True)
    parser.add_argument("--hard_train", type=str, required=True)
    parser.add_argument("--easy_val", type=str, required=True)
    parser.add_argument("--hard_val", type=str, required=True)
    parser.add_argument("--easy_test", type=str, required=True)
    parser.add_argument("--hard_test", type=str, required=True)

    # features list (comma-separated)
    parser.add_argument("--features", type=str, required=True)

    args = parser.parse_args()
    args.features = [f.strip() for f in args.features.split(",") if f.strip()]

    set_seed(args.seed)

    Xtr, ytr, Xva, yva, Xte, yte, scaler = load_splits(args)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    study.optimize(lambda t: objective(t, args, Xtr, ytr, Xva, yva), n_trials=args.trials)

    best = study.best_trial
    print("Best trial:", best.number, "val_auc=", best.value)
    print("Best params:", best.params)

    # Rebuild best model and evaluate on test once (optional sanity)
    hidden = tuple(best.params["hidden"])
    dropout = float(best.params["dropout"])
    lr = float(best.params["lr"])
    weight_decay = float(best.params["weight_decay"])
    batch_size = int(best.params["batch_size"])

    # retrain on train only with early stopping (or train+val if you want, but keep test sacred)
    device = args.device
    model = MLP(d=Xtr.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(RNADataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RNADataset(Xva, yva), batch_size=512, shuffle=False)
    test_loader = DataLoader(RNADataset(Xte, yte), batch_size=512, shuffle=False)

    best_state = None
    best_val_auc = -float("inf")
    no_improve = 0
    patience = int(best.params["patience"])
    min_delta = float(best.params["min_delta"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        pv, yv = eval_probs(model, val_loader, device)
        val_auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else float("nan")
        if np.isfinite(val_auc) and val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    pv, yv = eval_probs(model, val_loader, device)
    t_low, t_high = choose_ternary_thresholds(yv, pv, prec_bad=0.95, prec_good=0.95, min_gap=0.1)

    bad = pv < t_low
    good = pv >= t_high
    mid = ~(bad | good)
    print({"val_bad_rate": float(bad.mean()), "val_mid_rate": float(mid.mean()), "val_good_rate": float(good.mean())})
    print("Chosen thresholds:", t_low, t_high)

    pt, yt = eval_probs(model, test_loader, device)
    test_auc = roc_auc_score(yt, pt) if len(np.unique(yt)) > 1 else float("nan")
    print("Test AUC (one-time sanity):", test_auc)

    # Save a bundle (you can re-run your threshold picker afterward)
    bundle = {
        "meta": {"features": args.features, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
        "scaler": scaler,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_class": "MLP",
        "model_kwargs": {"d": int(Xtr.shape[1])},
        "arch": {"hidden": hidden, "dropout": dropout},
        "best_params": dict(best.params),
        "metrics": {"best_val_auc": float(best.value), "test_auc_sanity": float(test_auc)},
        "thresholds": {"low": float(t_low), "high": float(t_high)},
    }
    joblib.dump(bundle, args.bundle_out)
    print("Saved best bundle to:", args.bundle_out)


if __name__ == "__main__":
    main()
