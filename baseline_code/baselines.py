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

def fair_tress(X_train, y_train, X_test, prot_col_idx, seed):
    '''
    ref:
    Pereira Barata, A., Takes, F.W., van den Herik, H.J. et al. 
    Fair tree classifier using strong demographic parity. Mach Learn (2023). 
    https://doi.org/10.1007/s10994-023-06376-z

    https://link.springer.com/article/10.1007/s10994-023-06376-z
    https://github.com/pereirabarataap/fair_tree_classifier
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

    S_train = X_train.iloc[:, prot_col_idx]
    X_train.columns = range(X_train.shape[1])
    X_test.columns = range(X_test.shape[1])
    X_train = X_train.drop(prot_col_idx, axis=1)
    X_test = X_test.drop(prot_col_idx, axis=1)
    mdl = ft.FairRandomForestClassifier(random_state=seed, criterion='scaff')
    mdl.fit(X_train, y_train, S_train)
    preds = mdl.predict_proba(X_test)
    return mdl, preds

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

    return final_classification_model, y_pred


def zafar(X_train, y_train, X_test, prot_col_idx):
    '''
    ref: 

    Zafar, M.B., Valera, I., Rogriguez, M.G. & Gummadi, K.P.. (2017). 
    Fairness Constraints: Mechanisms for Fair Classification. 
    Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, 
    in Proceedings of Machine Learning Research 54:962-970 Available from https://proceedings.mlr.press/v54/zafar17a.html.
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

    pc = list(X_train.columns)[prot_col_idx]
    mdl, preds = z.fairness_constraints_paper(X_train, y_train, X_test, [pc])
    preds[preds==-1] = 0
    return mdl, preds    

def cfa(X_train, y_train, X_test, prot_col_idx, params):
    '''
    ref:

    Zhao, Yuying, Yu Wang, and Tyler Derr. 
    "Fairness and explainability: Bridging the gap towards fair model explanations." 
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 9. 2023.
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

    mdl = GridSearchCV(CFA(input_size=X_train.shape[1]), 
                params, 
                n_jobs=-1, 
                scoring='f1', 
                cv=3, error_score='raise')
    # TODO: Add back gridsearch if this works.
    mdl = CFA(input_size=X_train.shape[1], hidden_size=50, lr=.001, num_epochs=100, verbose=False)
    
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    preds = mdl.predict_proba(X_test).detach().numpy()[:, 1]
    return mdl, preds

def feldman(X_train, y_train, prot_col_idx): # If I can figure this out, extend to other AIF360 methods
    # TODO: Question for Nuno, there is no transform method only fit_transform, and the fit appears to require a data set with the ground truth
    # Therefore, it seems to me that we cannot pre-process the test data set and have to use the original features however IBM's docs seem
    # to suggest that we are supposed to be pre-processing both the train and test data (they first pre-process and then split train/test which is def wrong)
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

def gerryfair(X_train, y_train, X_test, y_test, prot_col_idx): # TODO: implement some type of cv?
    '''
    ref:
    Kearns, Michael, et al. 
    "Preventing fairness gerrymandering: Auditing and learning for subgroup fairness." 
    International conference on machine learning. PMLR, 2018.
    '''
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    X_train['y'] = y_train

    train_dataset = BinaryLabelDataset(df=X_train, 
                                    label_names=['y'], 
                                    protected_attribute_names=[X_train.columns[prot_col_idx]],
                                    favorable_label=1, # non-default
                                    unfavorable_label=0, # default label
                                )
    mdl = GerryFairClassifier()
    mdl.fit(train_dataset)

    X_test['y'] = y_test
    test_dataset = BinaryLabelDataset(df=X_test, 
                                    label_names=['y'], 
                                    protected_attribute_names=[X_test.columns[prot_col_idx]],
                                    favorable_label=1, # non-default
                                    unfavorable_label=0, # default label
                                )
    preds = mdl.predict(test_dataset).convert_to_dataframe()[0].y
    return mdl, preds


def prejudice_remover(X_train, y_train, X_test, y_test, prot_col_idx):
    '''
    ref:
    Kamishima, Toshihiro, et al. 
    "Fairness-aware classifier with prejudice remover regularizer." 
    Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2012, Bristol, UK, September 24-28, 2012. 
    Proceedings, Part II 23. Springer Berlin Heidelberg, 2012.
    '''
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    X_train['y'] = y_train

    train_dataset = BinaryLabelDataset(df=X_train, 
                                    label_names=['y'], 
                                    protected_attribute_names=[X_train.columns[prot_col_idx]],
                                    favorable_label=1, # non-default
                                    unfavorable_label=0, # default label
                                )
    mdl = PrejudiceRemover()
    mdl.fit(train_dataset)

    X_test['y'] = y_test
    test_dataset = BinaryLabelDataset(df=X_test, 
                                    label_names=['y'], 
                                    protected_attribute_names=[X_test.columns[prot_col_idx]],
                                    favorable_label=1, # non-default
                                    unfavorable_label=0, # default label
                                )
    preds = mdl.predict(test_dataset).convert_to_dataframe()[0].y
    return mdl, preds

def bayes_optimal_fair_preprocessing(): # FUDS - these are all weird
    # https://github.com/XianliZeng/Bayes-Optimal-Fair-Classification
    '''ref:
    Zeng, Xianli, Guang Cheng, and Edgar Dobriban. 
    "Bayes-optimal fair classification with linear disparity constraints via pre-, in-, and post-processing." 
    arXiv preprint arXiv:2402.02817 (2024).
    '''
    pass

def bayes_optimal_fair_inprocessing(): # FCSC
    '''ref:
    Zeng, Xianli, Guang Cheng, and Edgar Dobriban. 
    "Bayes-optimal fair classification with linear disparity constraints via pre-, in-, and post-processing." 
    arXiv preprint arXiv:2402.02817 (2024).
    '''
    pass

def bayes_optimal_fair_postprocessing(): # FPIR
    '''ref:
    Zeng, Xianli, Guang Cheng, and Edgar Dobriban. 
    "Bayes-optimal fair classification with linear disparity constraints via pre-, in-, and post-processing." 
    arXiv preprint arXiv:2402.02817 (2024).
    '''
    pass

def fair_bayes(X_train, y_train, X_test, prot_col_idx, model): # We be deleting this one fo sho
    # https://github.com/XianliZeng/FairBayes
    '''ref:
    Zeng, Xianli, Edgar Dobriban, and Guang Cheng. 
    "Bayes-optimal classifiers under group fairness." 
    arXiv preprint arXiv:2202.09724 (2022).
    '''
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_train, pd.DataFrame):
        X_train = np.array(X_train)

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(X_test, pd.DataFrame):
        X_test = np.array(X_test)

    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(y_train, pd.DataFrame):
        y_train = np.array(y_train)

    Z_train = X_train[:, prot_col_idx].clone() # Sensitve Attribute column
    preds_train = model.predict_proba(X_train)
    preds_train_1 = preds_train[Z_train==1]
    preds_train_0 = preds_train[Z_train==0]

    Z_test = X_test[:, prot_col_idx].clone()
    preds_test = model.predict_proba(X_test)
    preds_test_1 = preds_train[Z_test==1]
    preds_test_0 = preds_train[Z_test==0]

def fair_knn(X_train, y_train, X_test, prot_col_idx): # HAte it
    '''ref:
    Chakraborty, Joymallya, Kewen Peng, and Tim Menzies. 
    "Making fair ML software using trustworthy explanation." 
    Proceedings of the 35th IEEE/ACM International Conference on Automated Software Engineering. 2020.

    https://github.com/joymallyac/Fair-Knn/tree/master
    '''