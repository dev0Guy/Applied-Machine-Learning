import numpy as np
from sklearn.feature_selection import mutual_info_classif as MI
from sklearn.feature_selection import chi2 as CS
from ReliefF import ReliefF
from .whale_optimizer import WhaleOptimizer as WOA
from .utils import (
    score_feature_with_name,
    X_variance,
    score_knn_nb_svm,
    get_xgb_top_k,
    remove_correlated_features,
    get_features_by_bit_vector,
)
from sklearn.neighbors import KNeighborsClassifier

__all__ = ["PaperWrapper"]


class PaperWrapper:
    def _fittness_wrapper(self, X, y):
        def _fitness(sol):
            C = len(sol)
            acc_rate = 0
            R = len(sol[sol >= 0.5])
            if R > 0:
                sol_features = get_features_by_bit_vector(sol, self.feature_mapper)
                train = X.loc[:, sol_features].to_numpy()
                acc_rate += (
                    KNeighborsClassifier(n_neighbors=3).fit(train, y).score(train, y)
                )
            feature_part = self.Beta * (C - R / C)
            score_gamma = acc_rate * self.Alpha
            return score_gamma + feature_part

        return _fitness

    def __init__(self, M=50, K=60, J=23, R=0.7, Pop_n=70, Alpha=0.9, seed=123):
        self.M = M
        self.K = K
        self.J = J
        self.R = R
        self.Pop_n = Pop_n
        self.Alpha = Alpha
        self.Beta = 1 - self.Alpha
        self.bound = [0, 1]
        self.seed = seed

    def _ranked_union(self, X, y):
        # Get mutual information
        mi_top_m = [
            feature
            for feature, score in score_feature_with_name(X.columns, MI(X, y))[
                -self.M :
            ]
        ]
        # Get Chi Square
        cs_top_m = [
            feature
            for feature, score in score_feature_with_name(X.columns, CS(X, y)[1])[
                -self.M :
            ]
        ]
        # Get Xvariance
        cs_top_m = X_variance(X, y, self.M)
        # Get RFF
        relief_data = ReliefF(n_neighbors=20, n_features_to_keep=self.M).fit_transform(
            X.values, y.values
        )
        rff_top_m = []
        for col_name in X.columns:
            col_vector = X[col_name]
            for rff_vector in relief_data.T:
                if np.all(col_vector == rff_vector):
                    rff_top_m.append(col_name)
                    break
        # Union-set
        return set([*mi_top_m, *cs_top_m, *rff_top_m, *cs_top_m])

    def _phase_1(self, X, y):
        features_selected = self._ranked_union(X, y)
        score = score_knn_nb_svm(features_selected, X, y)
        top_k_features = [
            feature for feature, score in set(score_feature_with_name(X.columns, score))
        ]
        return get_xgb_top_k(top_k_features, X, y, self.J)

    def _phase_2(self, X, y, features):
        features = remove_correlated_features(X, y, features, self.R)
        return get_xgb_top_k(list(features), X, y, self.J)

    def _phase_3(self, X, y, features):
        self.feature_mapper = {idx: name for idx, name in enumerate(X.columns)}
        optimizer = WOA(
            self._fittness_wrapper(X, y),
            self.bound,
            self.Pop_n,
            len(features),
            self.seed,
            max_type=True,
        )
        solution = optimizer.run()
        return get_features_by_bit_vector(solution, self.feature_mapper)

    def __call__(self, X, y):
        features = self._phase_1(X, y)
        features = self._phase_2(X, y, features)
        return self._phase_3(X, y, features)
