import joblib
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d: int, hidden=(64, 32), dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


class QualityScorer:
    """
    Loads the joblib bundle once and scores feature dicts.
    """
    def __init__(self, bundle_path: str, device: str | None = None):
        self.bundle = joblib.load(bundle_path)

        self.features = list(self.bundle["meta"]["features"])
        self.scaler = self.bundle["scaler"]
        self.t_low = float(self.bundle["thresholds"]["low"])
        self.t_high = float(self.bundle["thresholds"]["high"])

        arch = self.bundle.get("arch", {})
        hidden = tuple(arch.get("hidden", (64, 32)))
        dropout = float(arch.get("dropout", 0.0))

        d = int(self.bundle["model_kwargs"]["d"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MLP(d=d, hidden=hidden, dropout=dropout).to(self.device)
        self.model.load_state_dict(self.bundle["model_state_dict"])
        self.model.eval()

    def _vectorize(self, attrs: dict) -> np.ndarray:
        missing = [f for f in self.features if f not in attrs]
        if missing:
            raise KeyError(f"Missing required features: {missing}")

        x = np.array([float(attrs[f]) for f in self.features], dtype=np.float32).reshape(1, -1)

        if not np.isfinite(x).all():
            bad_cols = [self.features[i] for i in np.where(~np.isfinite(x[0]))[0]]
            raise ValueError(f"Non-finite feature values for: {bad_cols}")

        return x

    @torch.no_grad()
    def score_attrs(self, attrs: dict) -> dict:
        x = self._vectorize(attrs)
        x = self.scaler.transform(x).astype(np.float32)

        xb = torch.from_numpy(x).to(self.device)
        logit = self.model(xb).item()
        prob_good = float(torch.sigmoid(torch.tensor(logit)).item())

        if prob_good < self.t_low:
            label = "probably bad"
        elif prob_good >= self.t_high:
            label = "probably good"
        else:
            label = "can't tell"

        return {
            "prob_good": prob_good,
            "label": label,
            "thresholds": {"low": self.t_low, "high": self.t_high},
            "logit": float(logit),
        }
