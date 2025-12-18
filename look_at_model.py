import joblib
from pprint import pprint

b = joblib.load("quality_model_bundle_optuna_best.joblib")

print("Top-level keys:", list(b.keys()))

print("\nBest params:")
pprint(b.get("best_params"))   # or b["best_params"] if you know it exists

print("\nArch:")
pprint(b.get("arch"))

print("\nThresholds:")
pprint(b.get("thresholds"))

print("\nMetrics:")
pprint(b.get("metrics"))

