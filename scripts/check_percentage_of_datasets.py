easy_path = "../chem_map_all_easy_preds_enriched.csv"
hard_path = "../chem_map_all_hard_preds_enriched.csv"

def eval_percentages(path):
    with open(path) as fh:
        data = fh.readlines()

    data.pop(0)
    total = len(data)
    datasets = {}
    data = [d.split(",") for d in data]
    for d in data:
        ds = d[0]
        if ds not in datasets.keys():
            datasets[ds] = 1
        else:
            datasets[ds] += 1

    for k, v in datasets.items():
        print(f"{k} - {v} ({((v / total) * 100):.2f}%)")


print("easy:")
eval_percentages(easy_path)
print("\nhard:")
eval_percentages(hard_path)
