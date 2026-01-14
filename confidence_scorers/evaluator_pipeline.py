from knn_scorer import KNNScorer
from gaussian_mm_scorer import GaussianMMScorer
from gaussian_nb_scorer import GaussianNBScorer
from log_reg_scorer import LogRegScorer
from svm_rbf_scorer import SVMRBFScorer


class Evaluator:
    def __init__(self):
        print(f"Initializing KNN scorer")
        knn_scorer = KNNScorer()
        print("Initializing Gaussian Naive Bayes scorer")
        gnb_scorer = GaussianNBScorer()
        print("Initializing Gaussian Mixture Model scorer")
        gmm_scorer = GaussianMMScorer()
        print("Initializing Logistic Regression scorer")
        lr_scorer = LogRegScorer()
        print("Initializing SVM-RBF scorer (might take a minute)")
        svm_scorer = SVMRBFScorer()

        self.models = {
            "KNN": knn_scorer,
            "Gaussian NB": gnb_scorer,
            "Gaussian MM": gmm_scorer,
            "Logistic Regression": lr_scorer,
            "SVM (RBF)": svm_scorer
        }

    def evaluate(self, seq, stc):
        for m_type, m_obj in self.models.items():
            msg = m_obj.interpret_score(seq, stc)
            print(f"{m_type}: {msg}")

