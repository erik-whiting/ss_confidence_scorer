import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from confidence_scorer import ConfidenceScorer


class GaussianMMScorer(ConfidenceScorer):
    def __init__(self, training_data_easy="../data/deduplicated_data/training_data_easy.csv", training_data_hard="../data/deduplicated_data/training_data_hard.csv"):
        super().__init__(training_data_easy=training_data_easy, training_data_hard=training_data_hard)
        Xe = self.easy_df[ConfidenceScorer.FEATURES].apply(pd.to_numeric, errors="coerce")
        Xh = self.hard_df[ConfidenceScorer.FEATURES].apply(pd.to_numeric, errors="coerce")

        Xe = Xe.loc[Xe.notna().all(axis=1)]
        Xh = Xh.loc[Xh.notna().all(axis=1)]

        self.scaler = StandardScaler()
        self.scaler.fit(pd.concat([Xe, Xh], ignore_index=True))

        Ze = self.scaler.transform(Xe)
        Zh = self.scaler.transform(Xh)

        # One GMM for each class (start with 1–3 components; 2 is a nice first try)
        self.gmm_easy = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        self.gmm_hard = GaussianMixture(n_components=2, covariance_type="full", random_state=0)

        self.gmm_easy.fit(Ze)
        self.gmm_hard.fit(Zh)


    def get_confidence_score(self, seq, stc):
        """
        Returns P(hard) based on Gaussian Naive Bayes (a likelihood-based classifier).
        """
        feature_dict = self.get_feature_dict(seq, stc)
        row = pd.DataFrame([feature_dict], columns=ConfidenceScorer.FEATURES).apply(pd.to_numeric, errors="coerce")
        if row.isna().any(axis=1).iloc[0]:
            missing = row.columns[row.isna().iloc[0]].tolist()
            raise ValueError(f"Missing/non-numeric features: {missing}")

        z = self.scaler.transform(row)

        ll_easy = float(self.gmm_easy.score_samples(z)[0])  # log p(x|easy)
        ll_hard = float(self.gmm_hard.score_samples(z)[0])  # log p(x|hard)

        # Convert log-likelihood ratio to (0,1) "hardness" score
        logit = ll_hard - ll_easy
        score = 1.0 / (1.0 + np.exp(-logit))
        return float(score)
