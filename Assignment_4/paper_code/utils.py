from functools import cmp_to_key
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import GradientBoostingClassifier
from statistics import mean
from sklearn.feature_selection import r_regression as PCC


def score_feature_with_name(columns_name, score):
    score = list(zip(columns_name, score))
    score = sorted(score, key=cmp_to_key(lambda item1, item2: item2[1] - item1[1]))
    return score


def X_variance(X, y, M):
    X_ = X[:].apply(pd.to_numeric, errors="coerce")
    # X_var.dtypes
    X_var = (X_).var()
    # print (X_var)
    Y_ = y[:].apply(pd.to_numeric, errors="coerce")
    Y_var = Y_.var()
    D = X_var + Y_var
    DL = D.nlargest(n=M)
    return np.array(DL.index)


def activate_model_on_features(features, X, y, model_class, model_kwargs):
    score_vector = np.zeros(X.shape[0])
    for idx, feature in enumerate(features):
        train = X[feature].to_numpy().reshape(1, -1).T
        test = y
        model = model_class(**model_kwargs).fit(train, test)
        score_vector[idx] = model.score(train, test)
    return score_vector


def score_knn_nb_svm(features_set, X, y):
    score = activate_model_on_features(
        features_set, X, y, KNeighborsClassifier, {"n_neighbors": 4}
    )
    score += activate_model_on_features(features_set, X, y, SVC, {"gamma": "auto"})
    score += activate_model_on_features(features_set, X, y, CategoricalNB, {})
    return score


def get_xgb_top_k(top_k_features, X, y, J):
    xgboost = GradientBoostingClassifier(
        n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
    ).fit(X, y)
    prev_accrucy = 0.0
    for idx in range(len(top_k_features)):
        k_selected_feature_names = top_k_features[: idx + 1]
        data = X.loc[:, k_selected_feature_names]
        current_accrucy = xgboost.fit(data, y).score(data, y)
        if len(k_selected_feature_names) >= J:
            break
    return top_k_features[:idx]


def remove_correlated_features(X, y, features, R):
    rmv_set = set()
    seen_set = set()
    class_val = y.to_numpy().reshape(1, -1).ravel()
    for _to, val1 in enumerate(features):
        if val1 in seen_set:
            continue
        for val2 in features[: _to + 1]:
            if val2 in seen_set or val2 in rmv_set:
                continue
            val1_data = X[val1].to_numpy().reshape(1, -1).T
            val2_data = X[val2].to_numpy()
            if abs(int(PCC(val1_data, val2_data))) >= R:
                val2_data = val2_data.reshape(1, -1).T
                rmv_val = (
                    val1
                    if abs(int(PCC(val1_data, class_val)))
                    > abs(int(PCC(val2_data, class_val)))
                    else val2
                )
                rmv_set.add(rmv_val)
            seen_set.add(val2)
        seen_set.add(val1)
    return set(features) - rmv_set


def get_features_by_bit_vector(vector, feature_mapper):
    feature_vec = []
    for idx, val in enumerate(vector):
        if val >= 0.5:
            feature_vec.append(feature_mapper[idx])
    return np.array(feature_vec)
