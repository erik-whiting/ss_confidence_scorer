import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from confidence_scorer import ConfidenceScorer


class KNNScorer(ConfidenceScorer):
    def __init__(self, training_data_easy="../data/deduplicated_data/training_data_easy.csv", training_data_hard="../data/deduplicated_data/training_data_hard.csv"):
        super().__init__(training_data_easy=training_data_easy, training_data_hard=training_data_hard)
        df = pd.concat([self.easy_df, self.hard_df], ignore_index=True)
        # Build X/y (numeric only, drop rows with missing feature values)
        X = df[ConfidenceScorer.FEATURES].apply(pd.to_numeric, errors="coerce")
        mask = X.notna().all(axis=1)
        X = X.loc[mask]
        y = df.loc[mask, "_label"]

        # kNN model (scaling is important for distance-based methods)
        self.knn = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15, weights="distance")  # tweak k later
        )
        self.knn.fit(X, y)


    def get_confidence_score(self, seq, stc):
        """
        Return P(hard) for a single datapoint represented as a dict of FEATURES -> value.
        """
        feature_dict = self.get_feature_dict(seq, stc)
        row = pd.DataFrame([feature_dict], columns=ConfidenceScorer.FEATURES).apply(pd.to_numeric, errors="coerce")
        if row.isna().any(axis=1).iloc[0]:
            missing = row.columns[row.isna().iloc[0]].tolist()
            raise ValueError(f"Missing/non-numeric features: {missing}")

        return float(self.knn.predict_proba(row)[0, 1])
