import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 1. Load and label the data
# ---------------------------

easy_path = "chem_map_all_easy_preds_enriched.csv"
hard_path = "chem_map_all_hard_preds_enriched.csv"

easy_df = pd.read_csv(easy_path)
hard_df = pd.read_csv(hard_path)

easy_df["label"] = 1
hard_df["label"] = 0

df = pd.concat([easy_df, hard_df], ignore_index=True)

# ---------------------------
# 2. Define feature columns
# ---------------------------

# Adjust names to EXACTLY match your CSV header
feature_cols = [
    # "score", # Not sure I want this
    "sequence_length",
    "gc_content",
    "sequence_entropy",
    "mfe",
    "ens_def",
    "longest_sequential_A",
    "longest_sequential_C",
    "longest_sequential_U",
    "longest_sequential_G",
    "longest_GC_helix",
    "GU_pairs",
    "rate_of_bps_predicted",
    "hairpin_count",
    "junction_count",
    "helix_count",
    "singlestrand_count",
    "mway_junction_count",
    "AU_pairs_in_helix_terminal_ends",
    "helices_with_reverse_complement",
    "hairpins_with_gt4_unpaired_nts",
]

# --------------------------------------
# 3. Force numeric + remove bad values
# --------------------------------------

# Make sure all feature columns are numeric
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with NaN in features or labels
df = df.dropna(subset=feature_cols + ["label"])

# Replace ±inf with NaN, then drop
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + ["label"])

# Ensure labels are 0 or 1
df["label"] = df["label"].astype(int)
assert set(df["label"].unique()) <= {0, 1}, "Labels must be 0/1 only."

# --------------------------------------
# 4. Remove constant (zero-variance) columns
# --------------------------------------
const_cols = []
for col in feature_cols:
    if df[col].nunique() <= 1:
        const_cols.append(col)

if const_cols:
    print("Dropping constant feature columns:", const_cols)
    feature_cols = [c for c in feature_cols if c not in const_cols]

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values.astype(np.int64)

# Quick sanity check
print("Any NaN in X?", np.isnan(X).any())
print("Any inf in X?", np.isinf(X).any())

# --------------------------------------
# 5. Train / val / test split
# --------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --------------------------------------
# 6. Scale features
# --------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Double-check after scaling
print("Any NaN in X_train_scaled?", np.isnan(X_train_scaled).any())
print("Any inf in X_train_scaled?", np.isinf(X_train_scaled).any())

# --------------------------------------
# 7. Dataset and DataLoaders
# --------------------------------------

class RNAPredictionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = RNAPredictionDataset(X_train_scaled, y_train)
val_dataset   = RNAPredictionDataset(X_val_scaled, y_val)
test_dataset  = RNAPredictionDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

# --------------------------------------
# 8. Model definition
# --------------------------------------

class QualityClassifier(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),   # single logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QualityClassifier(in_features=len(feature_cols)).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------------------------
# 9. Training / eval loop
# --------------------------------------

def run_epoch(loader, model, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()  # must be float for BCEWithLogitsLoss

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * X_batch.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        total_correct += (preds == y_batch.long()).sum().item()
        total_examples += X_batch.size(0)

    avg_loss = total_loss / total_examples
    avg_acc  = total_correct / total_examples
    return avg_loss, avg_acc


n_epochs = 20

for epoch in range(1, n_epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, model, optimizer)
    val_loss, val_acc     = run_epoch(val_loader, model, optimizer=None)

    print(
        f"Epoch {epoch:02d} | "
        f"train loss: {train_loss:.4f}, train acc: {train_acc:.3f} | "
        f"val loss: {val_loss:.4f}, val acc: {val_acc:.3f}"
    )

# --------------------------------------
# 10. Test evaluation
# --------------------------------------

test_loss, test_acc = run_epoch(test_loader, model, optimizer=None)
print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.3f}")
