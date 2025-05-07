from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.adversarial import AdversarialFairnessClassifier
import pandas as pd
import torch
import numpy as np
from baseline_code.CFA import CFA
import baseline_code.fair_trees as ft
import baseline_code.zafar as z
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE
import copy
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import GerryFairClassifier, PrejudiceRemover
from sklearn.model_selection import GridSearchCV


def agarwal(X_train, y_train, X_test, prot_col_idx, model):
    '''
    ref:

    Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J. & Wallach, H.. (2018). 
    A Reductions Approach to Fair Classification. Proceedings of the 35th International 
    Conference on Machine Learning, in Proceedings of Machine Learning Research 80:60-69 
    Available from https://proceedings.mlr.press/v80/agarwal18a.html.
    '''
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_train, pd.DataFrame):
        X_train = np.array(X_train)
    
    A_train = X_train[:, prot_col_idx]

    constraint = EqualizedOdds()
    mitigator = ExponentiatedGradient(model, constraint)
    mitigator.fit(X_train, y_train, sensitive_features=A_train)
    mitigation_preds = mitigator.predict(X_test)
    return mitigator, mitigation_preds

def hardt(X_train, y_train, X_test, prot_col_idx, model):
    '''
    ref:

    Hardt, Moritz, Eric Price, and Nati Srebro. 
    "Equality of opportunity in supervised learning." 
    Advances in neural information processing systems 29 (2016).
    '''
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    train_A = X_train.iloc[:, prot_col_idx]
    test_A = X_test.iloc[:, prot_col_idx]

    postproc_est = ThresholdOptimizer(estimator=model, constraints="equalized_odds", prefit=False, predict_method='predict')

    if (y_train == 1).sum() <= (y_train == 0).sum():
        balanced_idx1 = X_train[y_train == 1].index
        pp_train_idx = balanced_idx1.union(y_train[y_train == 0].sample(n=balanced_idx1.size).index)
    else:
        balanced_idx1 = X_train[y_train == 0].index
        pp_train_idx = balanced_idx1.union(y_train[y_train == 1].sample(n=balanced_idx1.size).index)
    X_train_balanced = X_train.loc[pp_train_idx, :]
    y_train_balanced = y_train.loc[pp_train_idx]
    A_train_balanced = train_A.loc[pp_train_idx]

    postproc_est.fit(X_train_balanced, y_train_balanced, sensitive_features=A_train_balanced)
    postproc_preds = postproc_est.predict(X_test, sensitive_features=test_A)
    return postproc_est, postproc_preds

def adv_deb(X_train, y_train, X_test, prot_col_idx, seed):
    '''
    ref:

    Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. 
    Mitigating unwanted biases with adversarial learning. 
    In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, 335–340. 2018.
    URL: https://dl.acm.org/doi/pdf/10.1145/3278721.3278779.
    '''
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    A_train = X_train.iloc[:, prot_col_idx]
    mitigator = AdversarialFairnessClassifier(
        backend="torch",
        predictor_model=[50, "leaky_relu"],
        adversary_model=[3, "leaky_relu"],
        batch_size=2 ** 8,
        progress_updates=0.5,
        random_state=seed,
    )
    mitigator.fit(X_train, y_train, sensitive_features=A_train)
    preds = mitigator.predict(X_test)
    preds = np.array(preds)
    return mitigator, preds

def xfair(X_train, y_train, X_test, prot_col_idx, seed):
    '''
    ref:

    Kewen Peng, Joymallya Chakraborty, and Tim Menzies. 2022. 
    FairMask: Better Fairness via Model-Based Rebalancing of Protected Attributes. 
    IEEE Trans. Softw. Eng. 49, 4 (April 2023), 2426–2439. https://doi.org/10.1109/TSE.2022.3220713
    '''
    def reg2clf(protected_pred, threshold=.5):
        out = []
        for each in protected_pred:
            if each >= threshold:
                out.append(1)
            else: out.append(0)
        return out
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    base_clf = RandomForestClassifier(random_state=seed, n_jobs=-1)  # This is from the code on github
    base2 = DecisionTreeRegressor(random_state=seed)

    final_classification_model = copy.deepcopy(base_clf)
    final_classification_model.fit(X_train, y_train)

    reduced = list(X_train.columns)
    reduced.remove(list(X_train.columns)[prot_col_idx])
    extrapolation_clfs = []
    
    X_reduced, y_reduced = X_train.loc[:, reduced], X_train.iloc[:, prot_col_idx]
    clf1 = copy.deepcopy(base2)
    sm = SMOTE()

    X_trains, y_trains = sm.fit_resample(X_reduced, y_reduced)
    if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
        clf1.fit(X_trains, y_trains)
    else:
        clf = copy.deepcopy(base_clf)
        clf.fit(X_trains, y_trains)
        y_proba = clf.predict_proba(X_trains)
        y_proba = [each[1] for each in y_proba]
        clf1.fit(X_trains, y_proba)

    extrapolation_clfs.append(clf1)

    X_test_reduced = X_test.loc[:, reduced]
    for i in range(len(extrapolation_clfs)):
        protected_pred = extrapolation_clfs[i].predict(X_test_reduced)
        if isinstance(extrapolation_clfs[i], DecisionTreeRegressor) or isinstance(extrapolation_clfs[i], LinearRegression):
            protected_pred = reg2clf(protected_pred, threshold=.5)
        X_test.iloc[:, prot_col_idx] = protected_pred

    y_pred = final_classification_model.predict_proba(X_test)

    return final_classification_model, y_pred[:, 1]

def feldman(X_train, y_train, prot_col_idx):
    '''
    ref:
    Feldman, Michael, et al. 
    "Certifying and removing disparate impact." 
    proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining. 2015.
    '''
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    X_train['y'] = y_train

    dataset = BinaryLabelDataset(df=X_train, 
                                    label_names=['y'], 
                                    protected_attribute_names=[X_train.columns[prot_col_idx]],
                                    favorable_label=1, # non-default
                                    unfavorable_label=0, # default label
                                )
    di_remover = DisparateImpactRemover(repair_level=.9)
    binary_label_dataset_transformed = di_remover.fit_transform(dataset)
    df_transformed = binary_label_dataset_transformed.convert_to_dataframe()[0]
    return df_transformed