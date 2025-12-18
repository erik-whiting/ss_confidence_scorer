def check_repeats(lines, kind):
    lines = [line.split(",") for line in lines]
    name_seq = [(line[1], line[2]) for line in lines]
    sequences = [line[2] for line in lines]
    repeated_data = []
    for name, seq in name_seq:
        count = sequences.count(seq)
        if count != 13:
            repeated_data.append([name, seq])
    print(f"Found {len(repeated_data)} of {len(lines)} in {kind}")


with open("scripts/training_data_easy.csv") as fh:
    lines = fh.readlines()
    lines.pop(0)
    check_repeats(lines, "easy")

with open("scripts/training_data_hard.csv") as fh:
    lines = fh.readlines()
    lines.pop(0)
    check_repeats(lines, "hard")

