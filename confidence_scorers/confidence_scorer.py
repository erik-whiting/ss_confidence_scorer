import pandas as pd
from sklearn.pipeline import make_pipeline

from attr_predictor import AttrPredictor


# Base class
class ConfidenceScorer:
    FEATURES = [
        "sequence_length","gc_content","sequence_entropy","mfe","ens_def",
        "longest_sequential_A","longest_sequential_C","longest_sequential_U","longest_sequential_G",
        "longest_GC_helix","GU_pairs","rate_of_bps_predicted","hairpin_count","junction_count",
        "helix_count","singlestrand_count","mway_junction_count","AU_pairs_in_helix_terminal_ends",
        "helices_with_reverse_complement","hairpins_with_gt4_unpaired_nts"
    ]

    def __init__(self,
                training_data_easy="../data/deduplicated_data/training_data_easy.csv",
                testing_data_easy  = "../data/deduplicated_data/easy_testing.csv",
                training_data_hard = "../data/deduplicated_data/training_data_hard.csv",
                testing_data_hard  = "../data/deduplicated_data/hard_testing.csv"
                ):
        self.easy_df = pd.read_csv(training_data_easy)
        self.hard_df = pd.read_csv(training_data_hard)
        self.easy_df["_label"] = 0  # 0 = easy
        self.hard_df["_label"] = 1  # 1 = hard


    def get_feature_dict(self, seq, stc):
        return AttrPredictor.pred_attrs(seq, stc)

    def get_confidence_score(self, seq, stc):
        raise NotImplementedError

    def interpret_score(self, seq, stc, hard_cutoff = 0.9, easy_cutoff = 0.1):
        score = self.get_confidence_score(seq, stc)
        score_string = f"{score:.5f}"
        if score >= hard_cutoff:
            return f"Prediction is likely inaccurate ({score_string})"
        elif score <= easy_cutoff:
            return f"Prediction is likely accurate ({score_string})"
        else:
            return f"Cannot determine confidence ({score_string})"


    def load_xy(self, easy, hard):
        easy_df = pd.read_csv(easy)
        hard_df = pd.read_csv(hard)

        easy_df["_label"] = 0  # easy
        hard_df["_label"] = 1  # hard

        self.df = pd.concat([easy_df, hard_df], ignore_index=True)
        X = self.df[ConfidenceScorer.FEATURES].apply(pd.to_numeric, errors="coerce")
        mask = X.notna().all(axis=1)
        X = X.loc[mask]
        y = self.df.loc[mask, "_label"]

        return X, y


    def help(self):
        print("A high score indicates that a given prediction for a sequence is likely not good")
        print("The features considered are:")
        for f in ConfidenceScorer.FEATURES:
            print(f"\t{f}")
