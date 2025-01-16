import EDOversampling
import MLP
import FairDataLoader
import FairnessMetrics as fm
import baseline_code.baselines as baselines
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import time
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import shap
import json
from aif360.datasets import BinaryLabelDataset
import logging
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from fastshap.utils import ShapleySampler
from torch.utils.data import DataLoader, TensorDataset


# TODO: Ask Nuno about discrepancy between proba_predict and predict (I try to use predict_proba when available but it appears that not all baselines offer this)

MLP_PARAM_SEARCH = {'hidden_size': [10, 50, 150],
                    'lr': [.001, .0001],
                    'num_epochs': [5], # [50, 100, 200],
                    'verbose': [False]}

def fastshap(X_train, mdl, num_epochs=50):
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
    imputer = lambda X, s: torch.tensor(mdl(X*s))
    if isinstance(X_train, np.ndarray) or isinstance(X_train, pd.DataFrame):
        X_train = torch.tensor(X_train, dtype=torch.float32)

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
    mdl = GridSearchCV(MLP.EDMLP(input_size=X_train.shape[1], output_size=1, seed=seed), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3, error_score='raise')
    start_time = time.time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['ED - MLP'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def ed_upsample(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING ED Upsampling")
    res = {}
    mdl = GridSearchCV(EDOversampling.EDOversampling(input_size=X_train.shape[1], output_size=1, verbose=False, seed=seed), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
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
    start_time = time.time()
    mdl.fit(X_train, y_train)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['MLP'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    
    logger.info("TRAINING AGARWAL")
    start_time = time.time()
    mdl = GridSearchCV(MLP.MLP(input_size=X_train.shape[1], output_size=1, verbose=False, batch_size=X_train.shape[0]), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.time()
    mdl.fit(X_train, y_train)
    agarwal_mdl, agarwal_preds = baselines.agarwal(X_train, y_train, X_test, prot_col_idx, mdl.best_estimator_)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    # e = shap.Explainer(agarwal_mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, agarwal_mdl.predict)
    res['Agarwal'] = evaluate(y_test, agarwal_preds, X_test, prot_col_idx, e, cols)

    logger.info("TRAINING HARDT")
    start_time = time.time()
    hardt_mdl, hardt_preds = baselines.hardt(X_train, np.array(y_train), X_test, prot_col_idx, mdl.best_estimator_)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')

    def predict_hardt(X1):
        d = pd.DataFrame(X1, columns=cols)
        return hardt_mdl.predict(d, sensitive_features=d['sex__Male'])
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
    start_time = time.time()
    X_train = baselines.feldman(X_train, y_train, prot_col_idx)
    y_train = X_train.y
    X_train = X_train.drop('y', axis=1)
    mdl.fit(X_train, y_train)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['Feldman'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)

    return res

def pr(X_train, y_train, X_test, y_test, prot_col_idx, cols):
    logger.info("TRAINING PREJUDICE REMOVER")
    res = {}
    start_time = time.time()
    mdl, preds = baselines.prejudice_remover(X_train, y_train, X_test, y_test, prot_col_idx)
    preds = np.array(preds)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
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
    start_time = time.time()
    mdl, preds = baselines.gerryfair(X_train, y_train, X_test, y_test, prot_col_idx)
    preds = np.array(preds)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
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
    start_time = time.time()
    mdl, preds = baselines.fair_tress(X_train, y_train, X_test, prot_col_idx, seed)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict_proba)
    res['Fair Trees'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def adv_deb(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING ADVERSARIAL DEBIASER")
    res = {}
    start_time = time.time()
    mdl, preds = baselines.adv_deb(X_train, y_train, X_test, prot_col_idx, seed)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict)
    res['Adversarial Debiasing'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def x_fair(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING XFAIR")
    res = {}
    start_time = time.time()
    mdl, preds = baselines.xfair(X_train, y_train, X_test, prot_col_idx, seed)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict)
    res['xFair'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def zafar(X_train, y_train, X_test, y_test, prot_col_idx, cols):
    logger.info("TRAINING ZAFAR")
    res = {}
    start_time = time.time()
    mdl, preds = baselines.zafar(X_train, y_train, X_test, prot_col_idx)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl)
    res['Zafar'] = evaluate(y_test, preds, X_test, prot_col_idx, e, cols)
    return res

def cfa(X_train, y_train, X_test, y_test, prot_col_idx, cols):
    logger.info("TRAINING CFA")
    res = {}
    start_time = time.time()
    mdl, preds = baselines.cfa(X_train, y_train, X_test, prot_col_idx, MLP_PARAM_SEARCH)
    logger.info(f'\tTime: {int(time.time() - start_time)//60} minutes')
    # e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=cols).iloc[:100, :])
    e = fastshap(X_train, mdl.predict)
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
    shap_vals = e(X)
    if hasattr(shap_vals, 'values'):
        shap_vals = shap_vals.values
    if isinstance(shap_vals, torch.Tensor):
        shap_vals.detach.numpy()
    # Normalize shaps and take abs value
    shap_vals = (np.apply_along_axis(np.divide, 0, abs(shap_vals), abs(shap_vals).sum(axis=1)))
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

def main():
    seed = 12345
    # logger.info(f"Start Time: {datetime.datetime.now()}")
    # np.random.seed(int(time.time()))
    # seed = np.random.randint(-2**31,2**31-1)+2**31
    # np.random.seed(seed)
    logger.info(f"SEED: {seed}")

    dataset = 'adult'
    if dataset == 'adult':
        X, y = FairDataLoader.get_adult_data()
        pc = 'sex__Male'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)
    scl = MinMaxScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    try:
        results = {}
        results.update(ed_mlp(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
        # results.update(ed_upsample(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
        logger.info(f"\n{pd.DataFrame(results)}")
        results.update(mlp_baselines(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
        logger.info(f"\n{pd.DataFrame(results)}")


        results.update(feldman(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns))
        # results.update(pr(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns))
        # results.update(gerryfair(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns))
        results.update(adv_deb(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
        results.update(fair_trees(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
        results.update(x_fair(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))
        logger.info(f"\n{pd.DataFrame(results)}")
        results.update(zafar(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns))
        logger.info(f"\n{pd.DataFrame(results)}")
        results.update(cfa(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns))

    finally:
        logger.info(results)
        with open('prelim_results.json', 'w') as fp:
            json.dump(results, fp)
        logger.info(f"\n{pd.DataFrame(results)}")


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
    main()