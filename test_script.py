from op_model import RNAPrediction

with open("chem_map_all_easy_preds_enriched.csv") as fh:
    data = fh.readlines()

data.pop(0)
data = [d.split(",") for d in data]

seq_pred = [(d[1], d[2], d[4]) for d in data]

target = "probably a good prediction"

for dp, seq, stc in seq_pred:
    result = RNAPrediction(seq, stc).get_confidence_score()
    if result["decision"] != target:
        print(f"{dp}: {result}")

