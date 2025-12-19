
def get_datapoints(input_path):
    with open(input_path) as fh:
        data = fh.readlines()

    data.pop(0)
    data = [d.split(",") for d in data]
    names_and_sequences = [(d[1], d[2]) for d in data]
    all_seqs = [ns[1] for ns in names_and_sequences]

    seq_dp_map = {}
    for seq in all_seqs:
        seq_dp_map[seq] = set()

    for name, seq in names_and_sequences:
        seq_dp_map[seq].add(name)

    dps_to_keep = set()
    for seq, dps in seq_dp_map.items():
        dps = list(dps)
        dps_to_keep.add(dps[0])

    print(f"Found {len(dps_to_keep)} datapoints to keep")
    master_dp = ",".join([item for item in dps_to_keep])
    return master_dp


def make_deduplicated_file(input_path, output_path):
    master_dps = get_datapoints(input_path)
    with open(input_path) as fh:
        data = fh.readlines()

    fstring = data.pop(0)
    data = [d.split(",") for d in data]

    for d in data:
        dp = d[1]
        if dp in master_dps:
            line = ",".join(d)
            fstring += line
    with open(output_path, "w") as fh:
        fh.write(fstring)



make_deduplicated_file("data/training_data_hard.csv", "data/deduplicated_data/training_data_hard.csv")
make_deduplicated_file("data/training_data_easy.csv", "data/deduplicated_data/training_data_easy.csv")
