import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from confidence_scorer import ConfidenceScorer


class SVMRBFScorer(ConfidenceScorer):
    def __init__(self, training_data_easy="../data/deduplicated_data/training_data_easy.csv", testing_data_easy="../data/deduplicated_data/easy_testing.csv", training_data_hard="../data/deduplicated_data/training_data_hard.csv", testing_data_hard="../data/deduplicated_data/hard_testing.csv"):
        super().__init__(training_data_easy, testing_data_easy, training_data_hard, testing_data_hard)
        X_train, y_train = self.load_xy(training_data_easy, training_data_hard)
        self.svm_rbf = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=10.0,            # good starting point; tune later
                gamma="scale",     # good default; tune later
                class_weight="balanced",
                probability=True,
                random_state=0,
            )
        )
        self.svm_rbf.fit(X_train, y_train)


    def get_confidence_score(self, seq, stc):
        feature_dict = self.get_feature_dict(seq, stc)
        """
        Return P(hard) for a single datapoint represented as a dict of FEATURES -> value.
        """
        row = pd.DataFrame([feature_dict], columns=ConfidenceScorer.FEATURES).apply(pd.to_numeric, errors="coerce")
        if row.isna().any(axis=1).iloc[0]:
            missing = row.columns[row.isna().iloc[0]].tolist()
            raise ValueError(f"Missing/non-numeric features: {missing}")
        return float(self.svm_rbf.predict_proba(row)[0, 1])
