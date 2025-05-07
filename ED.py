import pandas as pd
import numpy as np
import torch

def explanation_difference(e, X, prot_col_idx):
    if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame):
        X = torch.tensor(np.array(X), dtype=torch.float32)
    X_copy = X.clone()
    X_copy[:, prot_col_idx].apply_(lambda x: 1 if x==0 else 0 if x==1 else torch.nan)
    shaps = e(X)
    shaps_counter = e(X_copy)
    mean_diff = torch.mean(abs(shaps-shaps_counter),axis=0)
    return mean_diff.sum()

if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import MLP
    import FairDataLoader
    import torch.optim as optim
    import torch.nn as nn
    from fastshap.utils import ShapleySampler
    from torch.utils.data import DataLoader, TensorDataset

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

    MLP_PARAM_SEARCH = {'hidden_size': 10,
                    'lr': .001,
                    'num_epochs': 10,
                    'batch_size': 512,
                    'verbose': True}
    
    X, y = FairDataLoader.get_adult_data()
    pc = 'sex__Male'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)
    scl = MinMaxScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    prot_col_idx = list(X.columns).index(pc)    
    mdl = MLP.MLP_F1(input_size=X_train.shape[1], output_size=1, seed=12345, **MLP_PARAM_SEARCH)

    mdl.fit(X_train, y_train)
    preds = mdl.predict_proba(X_test)

    e = fastshap(X_train, mdl.predict_proba)
    print(explanation_difference(e, preds, X_test, prot_col_idx).item())