import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Callable, Optional


DEFAULT_HIDDEN = (64, 32)

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Optional[tuple] = None):
        super().__init__()
        h1, h2 = (hidden or DEFAULT_HIDDEN)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1),
        )
    def forward(self, x):  # returns logits
        return self.net(x).squeeze(1)

class QualityScorer:
    """
    Usage:
        extractor = lambda seq, dotbr: {... feature dict matching training FEATURES ...}
        scorer = QualityScorer("quality_model_bundle3.joblib", feature_extractor=extractor)
        result = scorer.score(sequence, dot_bracket)
        print(result)  # {'prob_good': 0.83, 'decision': 'probably good', ...}
    """
    def __init__(self, bundle_path: str, feature_extractor: Callable[[str, str], Dict[str, float]]):
        self.bundle = joblib.load(bundle_path)
        self.features = self.bundle["meta"]["features"]
        self.scaler = self.bundle["scaler"]
        self.t_low = self.bundle["thresholds"]["low"]
        self.t_high = self.bundle["thresholds"]["high"]

        mk = self.bundle.get("model_kwargs", {})
        in_dim = mk.get("d") or mk.get("in_dim") or len(self.features)
        hidden = mk.get("hidden", None)  # if you saved it; else None → DEFAULT_HIDDEN

        self.model = _MLP(in_dim, hidden=hidden)
        # state dict tensors might already be tensors; load as-is
        self.model.load_state_dict(self.bundle["model_state_dict"])
        self.model.eval()

        self.extract = feature_extractor

    def _prep_row(self, attrs: Dict[str, float]) -> np.ndarray:
        """Align attrs to training feature order and scale with the saved scaler."""
        row = pd.Series(attrs, dtype=float).reindex(self.features)

        # If any missing, impute with training means
        if row.isna().any():
            means = pd.Series(self.scaler.mean_, index=self.features)
            row = row.fillna(means)

        # Match how the scaler was originally fit
        if hasattr(self.scaler, "feature_names_in_"):
            assert list(self.features) == list(self.scaler.feature_names_in_), \
                "Feature names/order mismatch with saved scaler."
            X_new = pd.DataFrame([row.values], columns=self.features)
            Z = self.scaler.transform(X_new).astype(np.float32)
        else:
            Z = self.scaler.transform(row.to_numpy(dtype=float).reshape(1, -1)).astype(np.float32)

        return Z

    def score(self, sequence: str, prediction_dotbr: str) -> Dict[str, object]:
        """
        Returns:
          {
            'prob_good': float in [0,1],
            'decision': 'probably good' | 'probably bad' | 'cannot determine',
            'thresholds': {'low': t_low, 'high': t_high},
            'features_used': { ... attrs dict ... }
          }
        """
        attrs = self.extract(sequence, prediction_dotbr)  # must return all required features
        Z = self._prep_row(attrs)

        with torch.no_grad():
            logits = self.model(torch.from_numpy(Z))
            prob = torch.sigmoid(logits).cpu().numpy().ravel()[0].item()

        if prob >= self.t_high:
            decision = "probably good"
        elif prob < self.t_low:
            decision = "probably bad"
        else:
            decision = "cannot determine"

        return {
            "prob_good": float(prob),
            "decision": decision,
            "thresholds": {"low": self.t_low, "high": self.t_high},
            "features_used": attrs,
        }
