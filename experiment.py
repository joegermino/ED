import MLP
import EDMLP
import FairDataLoader
import FairnessMetrics as fm
import baseline_code.baselines as baselines
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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
from ED import explanation_difference

ED_MLP_PARAM_SEARCH = {'hidden_size': [10, 50, 150],
                    'lr': [.001, .0001],
                    'num_epochs': [50, 100, 250],
                    'batch_size': [128, 512, 1024],
                    'verbose': [False],
                    }

ED_MLP_ANYLOSS_PARAM_SEARCH = {'hidden_size': [10, 50, 150],
                    'lr': [.001, .0001],
                    'num_epochs': [50, 100, 250],
                    'L': [5, 25, 75],
                    'verbose': [False]
                    }

MLP_PARAM_SEARCH = {'hidden_size': [10, 50, 150],
                    'lr': [.001, .0001],
                    'num_epochs': [50, 100, 250],
                    'batch_size': [128, 512, 1024],
                    'L': [5, 25, 75],
                    'verbose': [False]
                    }

AGARWAL_PARAM_SEARCH = {'hidden_size': [10, 50, 150],
                    'lr': [.001, .0001],
                    'num_epochs': [50, 100, 250],
                    'verbose': [False]
                    }


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
    dataloader = DataLoader(dataloader, batch_size=256, shuffle=True)
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

def ed_mlp(X_train, y_train, X_test, y_test, prot_col_idx, seed):
    logger.info("TRAINING EDMLP")
    res = {}
    mdl = GridSearchCV(EDMLP.EDMLP(input_size=X_train.shape[1], output_size=1, seed=seed), 
                    ED_MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3, error_score='raise')
    ED_MLP_PARAM_SEARCH['input_size'] = [X_train.shape[1]]
    ED_MLP_PARAM_SEARCH['output_size'] = [1]

    start_time = time.process_time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    runtime = time.process_time() - start_time
    logger.info(f'\tTime: {runtime} seconds')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)

    e = fastshap(X_train, mdl.predict_proba)
    res['EDMLP'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['EDMLP']['Time'] = runtime
    logger.info(res)

    logger.info("TRAINING EDMLP with F1 Loss")
    mdl = GridSearchCV(EDMLP.EDMLP_F1(input_size=X_train.shape[1], output_size=1, seed=seed, batch_size=X_train.shape[0]), 
                    ED_MLP_ANYLOSS_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3, error_score='raise')
    start_time = time.process_time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    runtime = time.process_time() - start_time

    logger.info(f'\tTime: {runtime} seconds')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    e = fastshap(X_train, mdl.predict_proba)
    res['ED - MLP + F1 Loss'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['ED - MLP + F1 Loss']['Time'] = runtime
    logger.info(res)

    logger.info("TRAINING EDMLP with SP Loss")
    mdl = GridSearchCV(EDMLP.EDMLP_SP(input_size=X_train.shape[1], output_size=1, seed=seed, batch_size=X_train.shape[0]), 
                    ED_MLP_ANYLOSS_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3, error_score='raise')
    start_time = time.process_time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    runtime = time.process_time() - start_time

    logger.info(f'\tTime: {runtime} seconds')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    e = fastshap(X_train, mdl.predict_proba)
    res['ED - MLP + SP Loss'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['ED - MLP + SP Loss']['Time'] = runtime
    logger.info(res)

    logger.info("TRAINING EDMLP with F1 + SP Loss")
    mdl = GridSearchCV(EDMLP.EDMLP_F1_SP(input_size=X_train.shape[1], output_size=1, seed=seed, batch_size=X_train.shape[0]), 
                    ED_MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3, error_score='raise')
    start_time = time.process_time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    runtime = time.process_time() - start_time

    logger.info(f'\tTime: {runtime} seconds')
    logger.info(f'Best params: {mdl.best_params_}')
    logger.info(f'F1 Loss: {mdl.best_estimator_.f1_loss}')
    logger.info(f'SP Loss: {mdl.best_estimator_.sp_loss}')
    logger.info(f'ED Loss: {mdl.best_estimator_.ed_loss}')
    logger.info(f'FS Loss: {mdl.best_estimator_.fs_loss}')

    preds = mdl.predict_proba(X_test)
    e = fastshap(X_train, mdl.predict_proba)
    res['ED - MLP + F1 + SP Loss'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['ED - MLP + F1 + SP Loss']['Time'] = runtime
    logger.info(res)

    return res
    
def mlp_baselines(X_train, y_train, X_test, y_test, prot_col_idx, cols, seed):
    logger.info("TRAINING MLP F1")
    res = {}
    mdl = GridSearchCV(MLP.MLP_F1(input_size=X_train.shape[1], output_size=1, verbose=False, seed=seed), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    mdl.fit(X_train, y_train)
    runtime = time.process_time() - start_time
    logger.info(f'\tTime: {runtime} seconds')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    e = fastshap(X_train, mdl.predict_proba)
    res['MLP_F1'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['MLP_F1']['Time'] = runtime

    logger.info("TRAINING MLP SP")
    mdl = GridSearchCV(MLP.MLP_SP(input_size=X_train.shape[1], output_size=1, verbose=False, seed=seed), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    # mdl.fit(X_train, y_train)
    runtime = time.process_time() - start_time
    logger.info(f'\tTime: {runtime} seconds')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    e = fastshap(X_train, mdl.predict_proba)
    res['MLP_SP'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['MLP_SP']['Time'] = runtime

    logger.info("TRAINING MLP F1 + SP")
    mdl = GridSearchCV(MLP.MLP_F1_SP(input_size=X_train.shape[1], output_size=1, verbose=False, seed=seed), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    mdl.fit(X_train, y_train, **{'prot_col_idx': prot_col_idx})
    # mdl.fit(X_train, y_train)
    runtime = time.process_time() - start_time
    logger.info(f'\tTime: {runtime} seconds')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    e = fastshap(X_train, mdl.predict_proba)
    res['MLP_F1_SP'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['MLP_F1_SP']['Time'] = runtime
    
    logger.info("TRAINING AGARWAL")
    start_time = time.process_time()
    mdl = GridSearchCV(MLP.MLP_F1(input_size=X_train.shape[1], output_size=1, verbose=False, batch_size=X_train.shape[0]), 
                    AGARWAL_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    mdl.fit(X_train, y_train)
    agarwal_mdl, agarwal_preds = baselines.agarwal(X_train, y_train, X_test, prot_col_idx, mdl.best_estimator_)
    runtime = time.process_time() - start_time
    logger.info(f'\tTime: {runtime} seconds')
    e = fastshap(X_train, agarwal_mdl.predict)
    res['Agarwal'] = evaluate(y_test, agarwal_preds, X_test, prot_col_idx, e)
    res['Agarwal']['Time'] = runtime

    logger.info("TRAINING HARDT")
    start_time = time.process_time()
    hardt_mdl, hardt_preds = baselines.hardt(X_train, np.array(y_train), X_test, prot_col_idx, mdl.best_estimator_)
    runtime = time.process_time() - start_time
    logger.info(f'\tTime: {runtime} seconds')
    sf = cols[prot_col_idx]
    def predict_hardt(X1):
        d = pd.DataFrame(X1, columns=cols)
        return hardt_mdl.predict(d, sensitive_features=d[sf])
    e = fastshap(X_train, predict_hardt)
    res['Hardt'] = evaluate(y_test, hardt_preds, X_test, prot_col_idx, e)
    res['Hardt']['Time'] = runtime
    logger.info(res)

    return res

def feldman(X_train, y_train, X_test, y_test, prot_col_idx):
    logger.info("TRAINING FELDMAN")
    res = {}
    mdl = GridSearchCV(MLP.MLP_F1(input_size=X_train.shape[1], output_size=1, verbose=False), 
                    MLP_PARAM_SEARCH, 
                    n_jobs=-1, 
                    scoring='f1', 
                    cv=3)
    start_time = time.process_time()
    X_train = baselines.feldman(X_train, y_train, prot_col_idx)
    runtime = time.process_time() - start_time
    y_train = X_train.y
    X_train = X_train.drop('y', axis=1)
    mdl.fit(X_train, y_train)
    logger.info(f'\tTime: {runtime} seconds')
    logger.info(f'Best params: {mdl.best_params_}')
    preds = mdl.predict_proba(X_test)
    e = fastshap(X_train, mdl.predict_proba)
    res['Feldman'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['Feldman']['Time'] = runtime

    return res

def adv_deb(X_train, y_train, X_test, y_test, prot_col_idx, seed):
    logger.info("TRAINING ADVERSARIAL DEBIASER")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.adv_deb(X_train, y_train, X_test, prot_col_idx, seed)
    runtime = time.process_time() - start_time
    logger.info(f'\tTime: {runtime} seconds')
    e = fastshap(X_train, mdl.predict)
    res['Adversarial Debiasing'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['Adversarial Debiasing']['Time'] = runtime
    return res

def x_fair(X_train, y_train, X_test, y_test, prot_col_idx, seed):
    logger.info("TRAINING XFAIR")
    res = {}
    start_time = time.process_time()
    mdl, preds = baselines.xfair(X_train, y_train, X_test, prot_col_idx, seed)
    runtime = time.process_time() - start_time
    logger.info(f'\tTime: {runtime} seconds')
    def pred_xfair(X):
        return mdl.predict_proba(X)[:, 1]
    e = fastshap(X_train, pred_xfair)
    res['xFair'] = evaluate(y_test, preds, X_test, prot_col_idx, e)
    res['xFair']['Time'] = runtime
    return res

def evaluate(trues, preds, X, prot_col_idx, e):
    res = {}
    res['AUC'] = roc_auc_score(trues, preds)
    res['F1'] = f1_score(trues, (preds>=.5).astype(int))
    res['Accuracy'] = accuracy_score(trues, (preds>=.5).astype(int))
    res['Precision'] = precision_score(trues, (preds>=.5).astype(int))
    res['Recall'] = recall_score(trues, (preds>=.5).astype(int))
    res['Statistical Parity'] = fm.statistical_parity((preds>=.5).astype(int), X[:, prot_col_idx])
    res['Equalized Odds'] = fm.equalized_odds((preds>=.5).astype(int), np.array(trues), X[:, prot_col_idx])
    res['Explanation Difference'] = explanation_difference(e, X, prot_col_idx).item()
    return res

def main(label):
    np.random.seed(int(time.time()))
    seed = np.random.randint(-2**31,2**31-1)+2**31
    np.random.seed(seed)
    logger.info(f"SEED: {seed}")
    results = {}

    try:
        for dataset in ['adult', 'dutch_census', 'german_credit', 'bank_marketing', 'credit_card_clients', 'oulad', 'lawschool', 'compas', 'diabetes']: # Diabetes must be last! 'kdd_census', 
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
            elif dataset == 'folk_income':
                X, y = FairDataLoader.get_folk_income()
                pcs = ['SEX', 'RAC1P']
            elif dataset == 'folk_pc':
                X, y = FairDataLoader.get_folk_pc()
                pcs = ['SEX', 'RAC1P']
            elif dataset == 'folk_mobility':
                X, y = FairDataLoader.get_folk_mobility()
                pcs = ['SEX', 'RAC1P']
            elif dataset == 'folk_employment':
                X, y = FairDataLoader.get_folk_employment()
                pcs = ['SEX', 'RAC1P']
            elif dataset == 'folk_travel_time':
                X, y = FairDataLoader.get_folk_travel_time()
                pcs = ['SEX', 'RAC1P']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=seed)
            scl = MinMaxScaler()
            X_train = scl.fit_transform(X_train)
            X_test = scl.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            for pc in pcs:
                logger.info(f"PC: {pc.upper()}")
                results[dataset][pc] = {}
                results[dataset][pc].update(ed_mlp(X_train, y_train, X_test, y_test, list(X.columns).index(pc), seed))
                results[dataset][pc].update(mlp_baselines(X_train, y_train, X_test, y_test, list(X.columns).index(pc), X.columns, seed))

                results[dataset][pc].update(feldman(X_train, y_train, X_test, y_test, list(X.columns).index(pc)))
                results[dataset][pc].update(adv_deb(X_train, y_train, X_test, y_test, list(X.columns).index(pc),seed))
                results[dataset][pc].update(x_fair(X_train, y_train, X_test, y_test, list(X.columns).index(pc), seed))
                logger.info(f"\n{pd.DataFrame(results[dataset][pc])}")
                seed += 1
    finally:
        logger.info(results)
        with open(f'other_experiment_{label}.json', 'w') as fp:
            json.dump(results, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', default='')
    args = parser.parse_args()
    num_label = ''.join(filter(str.isdigit, args.label))
    sleep_time = 7*int(num_label) if len(num_label) > 0 else 0 # stall to diversify seed
    print(f"Stalling {sleep_time} seconds")
    time.sleep(sleep_time)
    logging.basicConfig(filename=f"log{args.label}.log",
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.INFO,
                    force=True)
    logger = logging.getLogger()
    pd.set_option('display.max_columns', 10)
    main(args.label)