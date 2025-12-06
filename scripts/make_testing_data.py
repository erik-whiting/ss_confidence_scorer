import os


easy_path = "validation_seperated_data_easy.csv"
hard_path = "validation_seperated_data_hard.csv"


targets = {
    "easy": 154 * 13,
    "hard": 154 * 13
}

def seperate_data(path, kind):
    with open(path) as fh:
        data = fh.readlines()

    headers = data.pop(0)
    data = [d.split(",") for d in data]

    eternapoints_captured = 0
    eterna_data = []
    new_data = []
    for d in data:
        if d[0] == "EternaData" and eternapoints_captured < targets[kind]:
            eternapoints_captured += 1
            eterna_data.append(d)
        else:
            new_data.append(d)

    # check
    most_datapoints = [d[1] for d in new_data]
    for d in eterna_data:
        if d[0] in most_datapoints:
            print(f"{d[1]} found in both sets ({kind} data)")
            break

    with open(f"{kind}_testing.csv", "w") as fh:
        fh.write(headers)
        for d in eterna_data:
            fh.write(",".join(d))

    with open(f"training_data_{kind}.csv", "w") as fh:
        fh.write(headers)
        for d in new_data:
            fh.write(",".join(d))



print("Easy:")
seperate_data(easy_path, "easy")

print("\nHard:")
seperate_data(hard_path, "hard")
