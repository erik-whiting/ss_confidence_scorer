# visualize_model.py
# Requires: pip install torchview graphviz joblib torch
# Also install Graphviz system binaries (e.g., brew install graphviz / apt-get install graphviz).

import joblib
import torch
import torch.nn as nn
from torchview import draw_graph

BUNDLE_PATH = "quality_model_bundle2.joblib"
OUT_BASENAME = "quality_model_architecture"  # will save OUT_BASENAME.png

# --- Define the same MLP shape you trained (hidden sizes must match) ---
HIDDEN = (64, 32)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, HIDDEN[0]), nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(),
            nn.Linear(HIDDEN[1], 1),
        )
    def forward(self, x):  # x: (batch, d)
        return self.net(x)

def main():
    bundle = joblib.load(BUNDLE_PATH)

    # Rebuild the model
    mk = bundle.get("model_kwargs", {})
    in_dim = mk.get("d") or mk.get("in_dim")  # support either key
    if in_dim is None:
        raise ValueError("Could not determine input dimension from bundle['model_kwargs'].")

    model = MLP(in_dim)
    state_dict = bundle["model_state_dict"]
    # (Optional) ensure tensors are on CPU
    state_dict = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Dummy input size: (batch=1, features=in_dim)
    graph = draw_graph(model, input_size=(1, in_dim), expand_nested=True)
    # Save as PNG (also supports "pdf", "svg")
    graph.visual_graph.render(OUT_BASENAME, format="png", cleanup=True)
    print(f"Saved diagram to {OUT_BASENAME}.png")

if __name__ == "__main__":
    main()
