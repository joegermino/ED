import EDOversampling
import MLP
import FairDataLoader
import FairnessMetrics as fm
import baseline_code.baselines as baselines
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import time
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import json
from aif360.datasets import BinaryLabelDataset
import logging
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from fastshap.utils import ShapleySampler
from torch.utils.data import DataLoader, TensorDataset

import threading

# TODO: Ask Nuno about discrepancy between proba_predict and predict (I try to use predict_proba when available but it appears that not all baselines offer this)

ED_MLP_PARAM_SEARCH = {'hidden_size': [10, 50, 150],
                    'lr': [.001, .0001],
                    'num_epochs': [50, 100, 200],
                    'batch_size': [128, 512, 1024],
                    'verbose': [False],
                    }

MLP_PARAM_SEARCH = {'hidden_size': [10, 50, 150],
                    'lr': [.001, .0001],
                    'num_epochs': [50, 100, 200],
                    'batch_size': [32, 128, 512],
                    'verbose': [False]}

LOCK = threading.Lock()

def thread(X_train, y_train, X_valid, y_valid, mdl, prot_attr_idx, params, seed, i, res):
    m = mdl(seed=seed, **params)
    m.fit(X_train, y_train, prot_attr_idx)
    preds = m.predict_proba(X_valid)
    f1 = f1_score(y_valid, (preds>=.5).astype(int))
    e = fastshap(X_train, m)
    ed = explanation_difference(e, preds, X_valid, prot_attr_idx)
    LOCK.acquire()
    res[i]['params'] = params
    res[i]['f1'] = f1
    res[i]['ed'] = ed
    LOCK.release()

def cv(X_train, y_train, params, seed, mdl, prot_attr_idx):
    thread_pool = []
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_t = X_train[train_idx, :]
        y_t = y_train[train_idx]
        X_v = X_train[valid_idx, :]
        y_v = X_train[valid_idx]
        param_pointers = {key: 0 for key in params.keys()}
        done = False
        idx = 0
        while not done:
            model_params = {key: params[key][param_pointers[key]] for key in params.keys()}
            i = 0
            param_pointers[list(params.keys())[i]] += 1
            while param_pointers[list(params.keys())[i]] == len(params[list(params.keys())[i]]):
                param_pointers[list(params.keys())[i]] = 0
                i += 1
                if i == len(params):
                    break
                param_pointers[list(params.keys())[i]] += 1
            res = {}
            idx += 1
            thread_pool.append(threading.Thread(target=thread, args=[X_t, y_t, X_v, y_v, mdl, prot_attr_idx, model_params, seed, idx, res]))
            if i == len(params):
                break
    for t in thread_pool:
        t.start()
    for t in thread_pool:
        t.join()
    print(res)

def fastshap(X_train, mdl, num_epochs=50, seed=12345):
    torch.manual_seed(seed)
    input_size = X_train.shape[1]
    explainer = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, input_size))
    optimizer = optim.Adam(explainer.parameters(), lr=.001)
    loss_fn = nn.MSELoss()
    S = ShapleySampler(X_train.shape[1])
    imputer = lambda X, s: torch.tensor(mdl(X*s), dtype=torch.float32)
    if isinstance(X_train, np.ndarray) or isinstance(X_train, pd.DataFrame):
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)

    dataloader = TensorDataset(X_train)
    dataloader = DataLoader(dataloader, batch_size=32, shuffle=True)
    for _ in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_data = [x for x in batch][0]
            temp_shaps = explainer(input_data)
            samp = S.sample(input_data.shape[0]*1, True)
            total = (temp_shaps * samp).sum(dim=1)
            grand = imputer(input_data, samp)
            null = imputer(input_data, 0)
            fs_loss = loss_fn(total, grand - null)
            fs_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
    return explainer

def ed_mlp(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING EDMLP")
    res = {}
    mdl = GridSearchCV(MLP.EDMLP(input_size=X_train.shape[1], output_size=1, seed=seed, lambda_=1), 
                    ED_MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3, error_score='raise')
    start_time = time.process_time()
    ED_MLP_PARAM_SEARCH['input_size'] = [X_train.shape[1]]
    ED_MLP_PARAM_SEARCH['output_size'] = [1]

    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)

    e = fastshap(X_train, mdl.predict_proba)
    res['ED - MLP'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    logger.info(res)
    logger.info("TRAINING EDMLP with SP Loss")
    mdl = GridSearchCV(MLP.EDMLP(input_size=X_train.shape[1], output_size=1, seed=seed, lambda_=1), 
                    ED_MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3, error_score='raise')
    start_time = time.process_time()
    ED_MLP_PARAM_SEARCH['input_size'] = [X_train.shape[1]]
    ED_MLP_PARAM_SEARCH['output_size'] = [1]

    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx, 'fair_loss': 'sp'})
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)

    e = fastshap(X_train, mdl.predict_proba)
    res['ED - MLP + SP Loss'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)

    logger.info("TRAINING EDMLP with EO Loss")
    mdl = GridSearchCV(MLP.EDMLP(input_size=X_train.shape[1], output_size=1, seed=seed, lambda_=1), 
                    ED_MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3, error_score='raise')
    start_time = time.process_time()
    ED_MLP_PARAM_SEARCH['input_size'] = [X_train.shape[1]]
    ED_MLP_PARAM_SEARCH['output_size'] = [1]

    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx, 'fair_loss': 'eo'})
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['ED - MLP + EO Loss'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    logger.info(res)
    return res

def ed_upsample(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING ED Upsampling")
    res = {}
    mdl = GridSearchCV(EDOversampling.EDOversampling(input_size=X_train.shape[1], output_size=1, verbose=False, seed=seed), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['ED - Upsample'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res
    
def mlp_baselines(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING MLP")
    res = {}
    mdl = GridSearchCV(MLP.MLP(input_size=X_train.shape[1], output_size=1, verbose=False, seed=seed), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    mdl.fit(X_train, y_train)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['MLP'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    
    logger.info("TRAINING AGARWAL")
    start_time = time.process_time()
    mdl = GridSearchCV(MLP.MLP(input_size=X_train.shape[1], output_size=1, verbose=False, batch_size=X_train.shape[0]), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    mdl.fit(X_train, y_train)
    agarwal_mdl, agarwal_preds = baselines.agarwal(X_train, y_train, X_test, prot_col_idx, mdl.best_estimator_)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    # e = shap.Explainer(agarwal_mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, agarwal_mdl.predict)
    res['Agarwal'] = evaluate(y_test, agarwal_preds, X_test, prot_col_idx, e, cols)

    logger.info("TRAINING HARDT")
    start_time = time.process_time()
    hardt_mdl, hardt_preds = baselines.hardt(X_train, np.array(y_train), X_test, prot_col_idx, mdl.best_estimator_)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    sf = cols[prot_col_idx]
    def predict_hardt(X1):
        d = pd.DataFrame(X1, columns=cols)
        return hardt_mdl.predict(d, sensitive_features=d[sf])
    # e = shap.Explainer(predict_hardt, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, predict_hardt)
    res['Hardt'] = evaluate(y_test, hardt_preds, X_test, prot_col_idx, e, cols)

    return res

def feldman(X_train, y_train, X_test, y_test, prot_col_idx, cols):
    logger.info("TRAINING FELDMAN")
    res = {}
    mdl = GridSearchCV(MLP.MLP(input_size=X_train.shape[1], output_size=1, verbose=False), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    X_train = baselines.feldman(X_train, y_train, prot_col_idx)
    y_train = X_train.y
    X_train = X_train.drop('y', axis=1)
    mdl.fit(X_train, y_train)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['Feldman'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)

    return res

def pr(X_train, y_train, X_test, y_test, prot_col_idx, cols):
    logger.info("TRAINING PREJUDICE REMOVER")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.prejudice_remover(X_train, y_train, X_test, y_test, prot_col_idx)
    preds = np.array(preds)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    X_test = pd.DataFrame(X_test, columns=cols)
    X_test['y'] = y_test
    ds = BinaryLabelDataset(df=X_test.iloc[:100, :], 
                                    label_names=['y'], 
                                    protected_attribute_names=[cols[prot_col_idx]],
                                    favorable_label=1, # non-default
                                    unfavorable_label=0, # default label
                                )
    # e = shap.Explainer(mdl.predict, ds)
    e = fastshap(X_train, mdl.predict) # This isn't going to work because of the nonsense with the BinaryLabelDatasets
    X_test = X_test.drop('y', axis=1)
    res['Prejudice Remover'] = evaluate(y_test, preds, np.array(X_test), prot_col_idx, e, cols)
    return res

def gerryfair(X_train, y_train, X_test, y_test, prot_col_idx, cols):
    logger.info("TRAINING GERRYFAIR")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.gerryfair(X_train, y_train, X_test, y_test, prot_col_idx)
    preds = np.array(preds)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    X_test = pd.DataFrame(X_test, columns=cols)
    X_test['y'] = y_test
    ds = BinaryLabelDataset(df=X_test.iloc[:100, :],   
                                    label_names=['y'], 
                                    protected_attribute_names=[cols[prot_col_idx]],
                                    favorable_label=1, # non-default
                                    unfavorable_label=0, # default label
                                )
    # e = shap.Explainer(mdl.predict, ds)
    e = fastshap(X_train, mdl.predict) # This isn't going to work because of the nonsense with the BinaryLabelDatasets
    res['Gerryfair'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def fair_trees(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING FAIR TREES")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.fair_trees(X_train, y_train, X_test, prot_col_idx, seed)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict)
    res['Fair Trees'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def adv_deb(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING ADVERSARIAL DEBIASER")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.adv_deb(X_train, y_train, X_test, prot_col_idx, seed)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict)
    res['Adversarial Debiasing'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def x_fair(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING XFAIR")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.xfair(X_train, y_train, X_test, prot_col_idx, seed)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    def pred_xfair(X):
        return mdl.predict_proba(X)[:, 1]
    e = fastshap(X_train, pred_xfair)
    res['xFair'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def zafar(X_train, y_train, X_test, y_test, prot_col_idx, cols):
    logger.info("TRAINING ZAFAR")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.zafar(X_train, y_train, X_test, prot_col_idx)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl)
    res['Zafar'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def cfa(X_train, y_train, X_test, y_test, prot_col_idx, cols):
    logger.info("TRAINING CFA")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.cfa(X_train, y_train, X_test, prot_col_idx, MLP_PARAM_SEARCH)
    logger.info(f'\tTime: {int(time.process_time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['CFA'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def evaluate(trues, preds, X, prot_col_idx, shap_vals, cols):
    res = {}
    res['AUC'] = roc_auc_score(trues, preds)
    res['F1'] = f1_score(trues, (preds>=.5).astype(int))
    res['Accuracy'] = accuracy_score(trues, (preds>=.5).astype(int))
    res['Precision'] = precision_score(trues, (preds>=.5).astype(int))
    res['Recall'] = recall_score(trues, (preds>=.5).astype(int))
    res['Statistical Parity'] = fm.statistical_parity((preds>=.5).astype(int), pd.DataFrame(X, columns=cols), list(cols)[prot_col_idx])
    res['Equalized Odds'] = fm.equalized_odds((preds>=.5).astype(int), pd.DataFrame(X, columns=cols), np.array(trues), list(cols)[prot_col_idx])
    res['Explanation Difference'] = explanation_difference(shap_vals, preds, X, prot_col_idx).item()
    return res

def explanation_difference(e, preds, X, prot_attr_idx):
    if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    shap_vals = e(X_tensor)
    if isinstance(shap_vals, torch.Tensor):
        shap_vals = shap_vals.detach().numpy()
    # Normalize shaps
    shap_vals = np.apply_along_axis(np.divide, 0, shap_vals - np.min(shap_vals, axis=1).reshape(-1,1), np.max(shap_vals, axis=1) - np.min(shap_vals, axis=1))
    a0 = (X[:, prot_attr_idx] == 0)
    a1 = (X[:, prot_attr_idx] == 1) 
    
    if shap_vals[a0 & (preds.flatten()>=.5), :].shape[0] == 0:
        s0_1_mean = np.zeros((X.shape[1]))
    else:
        s0_1_mean = np.mean(shap_vals[a0 & (preds.flatten()>=.5), :], axis=0) 
    if shap_vals[a1 & (preds.flatten()>=.5), :].shape[0] == 0:
        s1_1_mean = np.zeros((X.shape[1]))
    else:
        s1_1_mean = np.mean(shap_vals[a1 & (preds.flatten()>=.5), :], axis=0)
    
    if shap_vals[a0 & (preds.flatten()<.5), :].shape[0] == 0:
        s0_0_mean = np.zeros((X.shape[1]))
    else:
        s0_0_mean = np.mean(shap_vals[a0 & (preds.flatten()<.5), :], axis=0)
    if shap_vals[a1 & (preds.flatten()<.5), :].shape[0] == 0:
        s1_0_mean = np.zeros((X.shape[1]))
    else:
        s1_0_mean = np.mean(shap_vals[a1 & (preds.flatten()<.5), :], axis=0)
        
    return sum(abs(s0_1_mean - s1_1_mean) + abs(s0_0_mean - s1_0_mean))

def main(label):
    np.random.seed(int(time.process_time()))
    seed = np.random.randint(-2**31,2**31-1)+2**31
    np.random.seed(seed)
    logger.info(f"SEED: {seed}")
    results = {}

    try:
        for dataset in ['adult', 'dutch_census', 'german_credit', 'bank_marketing', 'credit_card_clients', 'oulad', 'lawschool', 'compas', 'kdd_census', 'diabetes']: # Diabetes must be last!
            logger.info(f"DATASET: {dataset}")
            results[dataset] = {}
            if dataset == 'adult':
                X, y = FairDataLoader.get_adult_data()
                pcs = ['sex__Male', 'race__White']
            elif dataset == 'dutch_census':
                X, y = FairDataLoader.get_dutch_census_data()
                pcs = ['sex_2']
            elif dataset == 'german_credit':
                X, y = FairDataLoader.get_german_credit_data()
                pcs = ['sex']
            elif dataset == 'bank_marketing':
                X, y = FairDataLoader.get_bank_marketing_data()
                pcs = ['marital_married']
            elif dataset == 'credit_card_clients':
                X, y = FairDataLoader.get_credit_card_data()
                pcs = ['SEX','MARRIAGE_2']
            elif dataset == 'oulad':
                X, y = FairDataLoader.get_oulad_data()
                pcs = ['gender']
            elif dataset == 'lawschool':
                X, y = FairDataLoader.get_lawschool_data()
                pcs = ['male', 'race']
            elif dataset == 'kdd_census':
                X, y = FairDataLoader.get_kdd_census_data()
                pcs = ['sex__Male', 'race__White']
            elif dataset == 'diabetes':
                X, y = FairDataLoader.get_diabetes_data()
                pcs = ['gender']
                ED_MLP_PARAM_SEARCH['batch_size'] = [256] # Because the diabetes data set is so small, a batch size of 128 breaks the code. 
                MLP_PARAM_SEARCH['batch_size'] = [256]
            elif dataset == 'compas':
                X, y = FairDataLoader.get_compas_data()
                pcs = ['race_Caucasian']
            else:
                raise ValueError("WTF")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)
            scl = MinMaxScaler()
            X_train = scl.fit_transform(X_train)
            X_test = scl.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            for pc in pcs:
                logger.info(f"PC: {pc.upper()}")
                results[dataset][pc] = {}
                results[dataset][pc].update(ed_mlp(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
                # # results[dataset][pc].update(ed_upsample(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
                logger.info(f"\n{pd.DataFrame(results[dataset][pc])}")
                results[dataset][pc].update(mlp_baselines(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
                logger.info(f"\n{pd.DataFrame(results[dataset][pc])}")

                results[dataset][pc].update(feldman(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns))
                # results[dataset][pc].update(pr(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns)) # BinaryLabelDataset incompatible with current FastSHAP code
                # results[dataset][pc].update(gerryfair(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns)) # BinaryLabelDataset incompatible with current FastSHAP code
                results[dataset][pc].update(adv_deb(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
                # results[dataset][pc].update(fair_trees(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed)) # Takes way too long at inference, explainer is unrealistic
                results[dataset][pc].update(x_fair(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
                logger.info(f"\n{pd.DataFrame(results[dataset][pc])}")
                # results[dataset][pc].update(zafar(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns)) # Takes way too long at inference, explainer is unrealistic
                # logger.info(f"\n{pd.DataFrame(results)}")
                # results[dataset][pc].update(cfa(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns))
                # seed += 1
    finally:
        logger.info(results)
        with open(f'full_results_lambda_1_run_{label}.json', 'w') as fp:
            json.dump(results, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', default='')
    args = parser.parse_args()
    logging.basicConfig(filename=f"log{args.label}.log",
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.INFO,
                    force=True)
    logger = logging.getLogger()
    pd.set_option('display.max_columns', 10)
    main(args.label)