with open("chem_map_easy_master_v5.csv") as fh:
    all_lines = fh.readlines()

all_lines.pop(0)

with open("chem_map_hard_master_v5.csv") as fh:
    lines = fh.readlines()


lines.pop(0)
all_lines += lines

all_dps = [line.split(",")[1] for line in all_lines]

with open("easy_and_hard_dps.txt", "w") as fh:
    fh.write("\n".join(all_dps))
