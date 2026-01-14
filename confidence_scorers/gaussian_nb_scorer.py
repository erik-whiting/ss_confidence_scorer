import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from confidence_scorer import ConfidenceScorer


class GaussianNBScorer(ConfidenceScorer):
    def __init__(self, training_data_easy="../data/deduplicated_data/training_data_easy.csv", training_data_hard="../data/deduplicated_data/training_data_hard.csv"):
        super().__init__(training_data_easy=training_data_easy, training_data_hard=training_data_hard)
        df = pd.concat([self.easy_df, self.hard_df], ignore_index=True)
        # Build X/y (numeric only, drop rows with missing feature values)
        X = df[ConfidenceScorer.FEATURES].apply(pd.to_numeric, errors="coerce")
        mask = X.notna().all(axis=1)
        X = X.loc[mask]
        y = df.loc[mask, "_label"]

        # Gaussian Naive Bayes
        self.density_model = make_pipeline(
            StandardScaler(),
            GaussianNB()
        )
        self.density_model.fit(X, y)


    def get_confidence_score(self, seq, stc):
        """
        Returns P(hard) based on Gaussian Naive Bayes (a likelihood-based classifier).
        """
        feature_dict = self.get_feature_dict(seq, stc)
        row = pd.DataFrame([feature_dict], columns=ConfidenceScorer.FEATURES).apply(pd.to_numeric, errors="coerce")
        if row.isna().any(axis=1).iloc[0]:
            missing = row.columns[row.isna().iloc[0]].tolist()
            raise ValueError(f"Missing/non-numeric features: {missing}")

        # Predict posterior via Bayes rule using class-conditional likelihoods
        return float(self.density_model.predict_proba(row)[0, 1])
