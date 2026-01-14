import math
from collections import Counter
from typing import Dict, Any, List

import vienna
from rna_secstruct import SecStruct
from RNAFoldAssess.utils.secondary_structure_tools import SecondaryStructureTools


class AttrPredictor:
    """
    Stateless helpers for computing sequence/structure-derived attributes.

    Notes:
    - `stc` is the dot-bracket (or whatever your predictor produces) for `seq`.
    - Motif-driven features require `motif_data` (list of motif strings like "HEL_seq_stc", etc.).
      If you don't have motif_data available inside pred_attrs(), those features default to 0.
    """

    @staticmethod
    def pred_attrs(seq: str, stc: str) -> Dict[str, Any]:
        fold_data = AttrPredictor.get_mfe_and_ens_def(seq)
        motif_data = []
        motifs_object = SecStruct(seq, stc)
        for motif in motifs_object.motifs.values():
            key = f"{motif.m_type}_{motif.sequence}_{motif.structure}"
            motif_data.append(key)

        attrs = {
            "sequence_length": len(seq),
            "gc_content": AttrPredictor.get_gc_content(seq),
            "sequence_entropy": AttrPredictor.calculate_entropy(seq),
            "mfe": fold_data["mfe"],
            "ens_def": fold_data["ens_def"],
            "longest_sequential_A": AttrPredictor.get_longest_nt(seq, "A"),
            "longest_sequential_C": AttrPredictor.get_longest_nt(seq, "C"),
            "longest_sequential_U": AttrPredictor.get_longest_nt(seq, "U"),
            "longest_sequential_G": AttrPredictor.get_longest_nt(seq, "G"),
            "longest_GC_helix": AttrPredictor.longest_gc_in_a_helix(motif_data),
            "GU_pairs": AttrPredictor.get_gu_pairs(seq, stc),
            "rate_of_bps_predicted": AttrPredictor.rate_of_bps_predicted(stc),
            "hairpin_count": AttrPredictor.count_motifs(motif_data, "HAIR"),
            "junction_count": AttrPredictor.count_motifs(motif_data, "JUNC"),
            "helix_count": AttrPredictor.count_motifs(motif_data, "HEL"),
            "singlestrand_count": AttrPredictor.count_motifs(motif_data, "SING"),
            "mway_junction_count": AttrPredictor.count_mway_junctions(motif_data),
            "AU_pairs_in_helix_terminal_ends": AttrPredictor.au_pairs_in_helix_ends(motif_data),
            "helices_with_reverse_complement": AttrPredictor.get_helices_with_reverse_comp_duplex(motif_data),
            "hairpins_with_gt4_unpaired_nts": AttrPredictor.count_hairpins_gt4_unpaired_nts(motif_data),
        }
        return attrs

    @staticmethod
    def get_gc_content(seq: str) -> float:
        if not seq:
            return 0.0
        gc = seq.count("C") + seq.count("G")
        return gc / len(seq)

    @staticmethod
    def calculate_entropy(seq: str) -> float:
        length = len(seq)
        if length == 0:
            return 0.0

        freqs = Counter(seq)
        probs = (count / length for count in freqs.values())
        return -sum(p * math.log(p, 4) for p in probs if p > 0)

    @staticmethod
    def get_mfe_and_ens_def(seq: str) -> Dict[str, float]:
        result = vienna.fold(seq)
        return {"mfe": result.mfe, "ens_def": result.ens_defect}

    @staticmethod
    def get_longest_nt(seq: str, nt: str) -> int:
        largest = 0
        for i in range(1, len(seq) + 1):
            if (nt * i) in seq:
                largest = i
            else:
                break
        return largest

    @staticmethod
    def longest_gc_in_a_helix(motif_data: List[str]) -> int:
        longest = 0
        helices = [h for h in motif_data if h.startswith("HEL")]
        for h in helices:
            _, hel_seq, hel_stc = h.split("_")
            hel_seq = hel_seq.replace("&", "")
            hel_stc = hel_stc.replace("&", "")

            pairings = SecondaryStructureTools.get_pairings(hel_seq, hel_stc)
            # normalize CG to GC for counting
            pairings = ["GC" if p in ("GC", "CG") else p for p in pairings]

            gc_run = 0
            best_run = 0
            for p in pairings:
                if p == "GC":
                    gc_run += 1
                    best_run = max(best_run, gc_run)
                else:
                    gc_run = 0

            longest = max(longest, best_run)

        return longest

    @staticmethod
    def get_gu_pairs(seq: str, stc: str) -> int:
        pairings = SecondaryStructureTools.get_pairings(seq, stc)
        return pairings.count("GU") + pairings.count("UG")

    @staticmethod
    def rate_of_bps_predicted(stc: str) -> float:
        if not stc:
            return 0.0
        bps = stc.count("(")
        if bps == 0:
            return 0.0
        return (bps * 2) / len(stc)

    @staticmethod
    def count_motifs(motif_data: List[str], mtype: str) -> int:
        return sum(1 for md in motif_data if md.startswith(mtype))

    @staticmethod
    def count_mway_junctions(motif_data: List[str]) -> int:
        count = 0
        junctions = [j for j in motif_data if j.startswith("JUNC")]
        for j in junctions:
            _, j_seq, _ = j.split("_")
            if j_seq.count("&") > 1:
                count += 1
        return count

    @staticmethod
    def au_pairs_in_helix_ends(motif_data: List[str]) -> float:
        helices = [h for h in motif_data if h.startswith("HEL")]
        if not helices:
            return 0.0
        total = sum(SecondaryStructureTools.get_au_helix_end_pairs(h) for h in helices)
        return total / len(helices)

    @staticmethod
    def get_helices_with_reverse_comp_duplex(motif_data: List[str]) -> float:
        helices = [h for h in motif_data if h.startswith("HEL")]
        if not helices:
            return 0.0
        count = sum(1 for h in helices if SecondaryStructureTools.helix_is_self_complementary_duplex(h))
        return count / len(helices)

    @staticmethod
    def count_hairpins_gt4_unpaired_nts(motif_data: List[str]) -> float:
        hairpins = [h for h in motif_data if h.startswith("HAIR")]
        if not hairpins:
            return 0.0
        count = sum(1 for h in hairpins if "...." in h)
        return count / len(hairpins)
