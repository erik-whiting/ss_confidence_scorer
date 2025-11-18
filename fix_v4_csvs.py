def fix_file(kind):
    with open(f"chem_map_{kind}_master_v4.csv") as fh:
        data = fh.readlines()

    header_row = data.pop(0)
    header_row = header_row.replace(",Unnamed: 0,", "")

    new_lines = []
    for d in data:
        d = d.split(",")
        new_line = ",".join(d[2:])
        new_lines.append(new_line)

    with open(f"chem_map_{kind}_master_v5.csv", "w") as fh:
        fh.write(header_row)
        fh.write("".join(new_lines))


fix_file("easy")
fix_file("hard")
