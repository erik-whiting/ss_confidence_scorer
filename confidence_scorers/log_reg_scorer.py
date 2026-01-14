import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from confidence_scorer import ConfidenceScorer


class LogRegScorer(ConfidenceScorer):
    def __init__(self,
                 training_data_easy="../data/deduplicated_data/training_data_easy.csv",
                testing_data_easy  = "../data/deduplicated_data/easy_testing.csv",
                training_data_hard = "../data/deduplicated_data/training_data_hard.csv",
                testing_data_hard  = "../data/deduplicated_data/hard_testing.csv"
                ):
        super().__init__(training_data_easy, testing_data_easy, training_data_hard, testing_data_hard)
        X_train, y_train = self.load_xy(training_data_easy, training_data_hard)
        X_test,  y_test  = self.load_xy(testing_data_easy,  testing_data_hard)
        self.lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                class_weight="balanced",  # good default if easy/hard sizes differ
            )
        )
        self.lr.fit(X_train, y_train)


    def get_confidence_score(self, seq, stc):
        feature_dict = self.get_feature_dict(seq, stc)
        row = pd.DataFrame([feature_dict], columns=ConfidenceScorer.FEATURES).apply(pd.to_numeric, errors="coerce")
        if row.isna().any(axis=1).iloc[0]:
            missing = row.columns[row.isna().iloc[0]].tolist()
            raise ValueError(f"Missing/non-numeric features: {missing}")
        return float(self.lr.predict_proba(row)[0, 1])
