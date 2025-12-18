from op_model import RNAPrediction

with open("scripts/easy_testing.csv") as fh:
    lines = fh.readlines()

lines.pop(0)
dp_count = len(lines)
wrong_count = 0
for line in lines:
    line = line.split(",")
    seq = line[2]
    stc = line[4]
    rp = RNAPrediction(seq, stc)
    result = rp.get_confidence_score()
    label = result["label"]
    if label != "probably good":
        wrong_count += 1

results_text = open("test_results.txt", "w")

rate = wrong_count / dp_count
results_text.write(f"Model is wrong on easy data {(rate * 100):2f}% of the time\n")

with open("scripts/hard_testing.csv") as fh:
    lines = fh.readlines()

lines.pop(0)
dp_count = len(lines)
wrong_count = 0
for line in lines:
    line = line.split(",")
    seq = line[2]
    stc = line[4]
    rp = RNAPrediction(seq, stc)
    result = rp.get_confidence_score()
    label = result["label"]
    if label != "probably bad":
        wrong_count += 1

rate = wrong_count / dp_count
results_text.write(f"Model is wrong on hard data {(rate * 100):2f}% of the time")
results_text.close()

