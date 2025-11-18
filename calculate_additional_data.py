import os


from rna_secstruct import SecStruct
from RNAFoldAssess.utils.secondary_structure_tools import SecondaryStructureTools


"""
Additional attributes

Longest sequential A,C,U,G
Longest GC helix
GU pairs
Rate of basepairs predicted
hairpin count
junction count
helix count
singlestrand count
multiway junction count
AU pairs in helices
helices with reverse complement
rate of > 4 unapired nucleotides in hairpin
"""

def restructure_data(kind):
    with open(f"chem_map_all_{kind}_preds.csv") as fh:
        lines = fh.readlines()
    headers = lines.pop(0).strip()
    headers += ",longest_sequential_A,longest_sequential_C,longest_sequential_U,longest_sequential_G,"
    headers += "longest_GC_helix,GU_pairs,rate_of_bps_predicted,"
    headers += "hairpin_count,junction_count,helix_count,singlestrand_count,"
    headers += "mway_junction_count,AU_pairs_in_helix_terminal_ends,helices_with_reverse_complement,hairpins_with_gt4_unpaired_nts\n"

    new_file = open(f"chem_map_all_{kind}_preds_enriched.csv", "w")
    new_file.write(headers)

    lines = [line.strip().split(",") for line in lines]
    for line in lines:
        # Prepare new line
        new_line = ",".join(line) + ","

        seq = line[2]
        stc = line[4]
        mdata = get_motif_data(seq, stc)

        for nt in "ACUG":
            count = get_longest_nt(seq, nt)
            new_line += f"{count},"

        helices = [m for m in mdata if "HELIX" in m]
        longest_gc = longest_gc_in_a_helix(helices)
        gu_pairs = get_gu_pairs(seq, stc)
        bp_rate = rate_of_bps_predicted(stc)
        new_line += f"{longest_gc},{gu_pairs},{bp_rate},"

        for mtype in ["HAIRPIN", "JUNCTION", "HELIX", "SINGLESTRAND"]:
            motif_count = get_motif_count(mdata, mtype)
            new_line += f"{motif_count},"

        junctions = [m for m in mdata if "JUNCTION" in m]
        mway_count = count_mway_junctions(junctions)
        au_pairs = au_pairs_in_helix_ends(helices)
        h_revcomp = get_helices_with_reverse_comp_duplex(helices)
        hairpins = [m for m in mdata if "HAIRPIN" in m]
        hp_with_gt4 = count_hairpins_gt4_unpaired_nts(hairpins)
        new_line += f"{mway_count},{au_pairs},{h_revcomp},{hp_with_gt4}\n"

        new_file.write(new_line)

    new_file.close()



def get_motif_data(seq, stc):
    mdata = []
    motifs_object = SecStruct(seq, stc)
    for motif in motifs_object.motifs.values():
        key = f"{motif.m_type}_{motif.sequence}_{motif.structure}"
        mdata.append(key)
    return mdata


def get_longest_nt(seq, nt):
    largest = 0
    for i in range(1, len(seq)):
        needle = nt * i
        if needle in seq:
            largest = i
        else:
            break
    return largest


def get_motif_count(motif_data, motif):
    return len([m for m in motif_data if m.startswith(motif)])


def longest_gc_in_a_helix(helices):
    longest = 0
    for h in helices:
        _, seq, stc = h.split("_")
        seq = seq.replace("&", "")
        stc = stc.replace("&", "")
        pairings = SecondaryStructureTools.get_pairings(seq, stc)
        new_pairings = []
        for p in pairings:
            if p == "CG":
                new_pairings.append("GC")
            else:
                new_pairings.append(p)

        pairings = new_pairings

        gc_pair_length = 0

        for p in pairings:
            if p == "GC":
                gc_pair_length += 1
            else:
                gc_pair_length = 0

        if gc_pair_length > longest:
            longest = gc_pair_length
    return longest

def get_gu_pairs(seq, stc):
    pairings = SecondaryStructureTools.get_pairings(seq, stc)
    return pairings.count("GU") + pairings.count("UG")


def rate_of_bps_predicted(stc):
    bps = stc.count("(")
    if bps == 0:
        return 0
    return (bps * 2) / len(stc)


def count_motifs(motif_data, motif_type):
    count = 0
    for md in motif_data:
        if md.startswith(motif_type):
            count += 1
    return count


def count_mway_junctions(junctions):
    count = 0
    for j in junctions:
        _, seq, stc = j.split("_")
        if seq.count("&") > 1:
            count += 1
    return count


def au_pairs_in_helix_ends(helices):
    if len(helices) == 0:
        return 0
    all_count = 0
    for h in helices:
        count = SecondaryStructureTools.get_au_helix_end_pairs(h)
        all_count += count
    return all_count / len(helices)


def get_helices_with_reverse_comp_duplex(helices):
    if len(helices) == 0:
        return 0
    count = 0
    for h in helices:
        if SecondaryStructureTools.helix_is_self_complementary_duplex(h):
            count += 1

    return count / len(helices)


def count_hairpins_gt4_unpaired_nts(hairpins):
    if len(hairpins) == 0:
        return 0
    count = 0
    for h in hairpins:
        if "...." in h:
            count += 1

    return count / len(hairpins)


if __name__ == "__main__":
    print("working hard cases")
    restructure_data("hard")
    print("working easy cases")
    restructure_data("easy")
