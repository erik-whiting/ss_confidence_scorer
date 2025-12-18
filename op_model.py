import math
from collections import Counter

import vienna
from rna_secstruct import SecStruct
from RNAFoldAssess.utils.secondary_structure_tools import SecondaryStructureTools
from quality_model_runtime import QualityScorer


class RNAPrediction:
    _scorer = None

    def __init__(self, sequence, prediction):
        self.sequence = sequence
        self.prediction = prediction
        self.motif_data = []
        motifs_object = SecStruct(self.sequence, self.prediction)
        for motif in motifs_object.motifs.values():
            key = f"{motif.m_type}_{motif.sequence}_{motif.structure}"
            self.motif_data.append(key)

    def get_confidence_score(self):
        if RNAPrediction._scorer is None:
            RNAPrediction._scorer = QualityScorer("quality_model_bundle_optuna_best.joblib")

        attrs = self.pred_attrs()
        return RNAPrediction._scorer.score_attrs(attrs)


    def pred_attrs(self):
        fold_data = self.get_mfe_and_ens_def()
        attrs = {
            "sequence_length": len(self.sequence),
            "gc_content": self.get_gc_content(),
            "sequence_entropy": self.calculate_entropy(),
            "mfe": fold_data["mfe"],
            "ens_def": fold_data["ens_def"],
            "longest_sequential_A": self.get_longest_nt("A"),
            "longest_sequential_C": self.get_longest_nt("C"),
            "longest_sequential_U": self.get_longest_nt("U"),
            "longest_sequential_G": self.get_longest_nt("G"),
            "longest_GC_helix": self.longest_gc_in_a_helix(),
            "GU_pairs": self.get_gu_pairs(),
            "rate_of_bps_predicted": self.rate_of_bps_predicted(),
            "hairpin_count": self.count_motifs("HAIR"),
            "junction_count": self.count_motifs("JUNC"),
            "helix_count": self.count_motifs("HEL"),
            "singlestrand_count": self.count_motifs("SING"),
            "mway_junction_count": self.count_mway_junctions(),
            "AU_pairs_in_helix_terminal_ends": self.au_pairs_in_helix_ends(),
            "helices_with_reverse_complement": self.get_helices_with_reverse_comp_duplex(),
            "hairpins_with_gt4_unpaired_nts": self.count_hairpins_gt4_unpaired_nts(),
        }
        return attrs


    def get_gc_content(self):
        gc = self.sequence.count("C") + self.sequence.count("G")
        return gc / len(self.sequence)

    def calculate_entropy(self):
        length = len(self.sequence)
        if length == 0:
            return 0.0

        freqs = Counter(self.sequence)
        probs = [count / length for count in freqs.values()]
        entropy = -sum(p * math.log(p, 4) for p in probs if p > 0)

        return entropy

    def get_mfe_and_ens_def(self):
        result = vienna.fold(self.sequence)
        return {"mfe": result.mfe, "ens_def": result.ens_defect}

    def get_longest_nt(self, nt):
        largest = 0
        for i in range(1, len(self.sequence)):
            needle = nt * i
            if needle in self.sequence:
                largest = i
            else:
                break
        return largest

    def longest_gc_in_a_helix(self):
        longest = 0
        helices = [h for h in self.motif_data if h.startswith("HEL")]
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

    def get_gu_pairs(self):
        pairings = SecondaryStructureTools.get_pairings(self.sequence, self.prediction)
        return pairings.count("GU") + pairings.count("UG")

    def rate_of_bps_predicted(self):
        bps = self.prediction.count("(")
        if bps == 0:
            return 0
        return (bps * 2) / len(self.prediction)

    def count_motifs(self, mtype):
        count = 0
        for md in self.motif_data:
            if md.startswith(mtype):
                count += 1
        return count

    def count_mway_junctions(self):
        count = 0
        junctions = [j for j in self.motif_data if j.startswith("JUNC")]
        for j in junctions:
            _, seq, stc = j.split("_")
            if seq.count("&") > 1:
                count += 1
        return count

    def au_pairs_in_helix_ends(self):
        helices = [h for h in self.motif_data if h.startswith("HEL")]
        if len(helices) == 0:
            return 0
        all_count = 0
        for h in helices:
            count = SecondaryStructureTools.get_au_helix_end_pairs(h)
            all_count += count
        return all_count / len(helices)

    def get_helices_with_reverse_comp_duplex(self):
        helices = [h for h in self.motif_data if h.startswith("HEL")]
        if len(helices) == 0:
            return 0
        count = 0
        for h in helices:
            if SecondaryStructureTools.helix_is_self_complementary_duplex(h):
                count += 1

        return count / len(helices)

    def count_hairpins_gt4_unpaired_nts(self):
        hairpins = [h for h in self.motif_data if h.startswith("HAIR")]
        if len(hairpins) == 0:
            return 0
        count = 0
        for h in hairpins:
            if "...." in h:
                count += 1

        return count / len(hairpins)
