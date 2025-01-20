import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from EDLoss import EDLoss
from SPLoss import SPLoss
from EOLoss import EOLoss
from fastshap.utils import ShapleySampler

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32):
        super(MLP, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def get_params(self, deep=False):
        return {'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'verbose': self.verbose,
                'seed': self.seed,
                'batch_size': self.batch_size
               }
    
    def fit(self, X, y, sample_weight=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)
        if sample_weight is not None:
            if type(sample_weight) != torch.Tensor:
                sample_weight = torch.tensor(np.array(sample_weight))
            criterion = nn.BCEWithLogitsLoss(weight=sample_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        for _ in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            self._train_epoch(dataloader, optimizer, criterion)
    
    def _train_epoch(self, iterator, optimizer, criterion):
        self.train()
        epoch_loss = 0
        for batch in iterator:
            optimizer.zero_grad()
            input_data, labels = [x for x in batch]
            output = self(input_data)
            loss = criterion(output.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
            
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        dataloader = TensorDataset(X)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=False) 
        self.eval()
        preds = np.array([])
        with torch.no_grad():
            for inputs in dataloader:
                batch_preds = self._predict_batch(inputs[0])
                preds = np.append(preds, batch_preds)
        return (preds >=.5).astype(int)
    
    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        dataloader = TensorDataset(X)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=False) 
        self.eval()
        preds = np.array([])
        with torch.no_grad():
            for inputs in dataloader:
                batch_preds = self._predict_batch(inputs[0])
                preds = np.append(preds, batch_preds)
        return preds
    
    def _predict_batch(self, X):
        self.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        with torch.no_grad():
            logits = self(X)
            preds = torch.sigmoid(logits)
            preds = preds.detach().numpy().flatten()
            return preds
        
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=False, batch_size=None):
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if lr is not None:
            self.lr = lr
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if input_size is not None:
            self.input_size = input_size
        if output_size is not None:
            self.output_size = output_size
        if verbose is not None:
            self.verbose = verbose
        if batch_size is not None:
            self.batch_size = batch_size
        return self

class EDMLP(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, lambda_=1):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        self.batch_size = batch_size
        torch.manual_seed(self.seed)
        super(EDMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
             nn.Linear(hidden_size, output_size)   
        )
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.lambda_ = lambda_
        self.prot_col_idx = None

        self.explainer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_size))

    def forward(self, x):
        return self.mlp(x)
    
    def set_prot_col_idx(self, i):
        self.prot_col_idx = i
    
    def fit(self, X, y, prot_col_idx=None, sample_weight=None, fair_loss=None):
        if prot_col_idx is None and self.prot_col_idx is None:
            raise ValueError("prot_col_idx cannot be null")
        if prot_col_idx is not None:
            self.prot_col_idx = prot_col_idx
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        self.fs_losses = []
        self.fair_losses = []
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)

        fs_optimizer = optim.Adam(self.explainer.parameters(), lr=self.lr)
        mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=self.lr)
        criterion_fair = EDLoss()
        if sample_weight is not None:
            if type(sample_weight) != torch.Tensor:
                sample_weight = torch.tensor(np.array(sample_weight))
            criterion_perf = nn.BCEWithLogitsLoss(weight=sample_weight)
        else:
            criterion_perf = nn.BCEWithLogitsLoss()
        if fair_loss == 'sp':
            criterion_standard_fair = SPLoss()
        elif fair_loss == 'eo':
            criterion_standard_fair = EOLoss()
        else:
            criterion_standard_fair = None
        loss_fn = nn.MSELoss()
        S = ShapleySampler(X.shape[1])
        for _ in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            self._train_epoch(dataloader, criterion_fair, criterion_perf, criterion_standard_fair, fs_optimizer, mlp_optimizer, loss_fn, S, self.prot_col_idx, fair_loss)
        
    def _train_epoch(self, iterator, criterion_fair, criterion_perf, criterion_standard_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col, fair_loss):
        self.train()
        imputer = lambda X, s: torch.sigmoid(self(X*s))
        for batch in iterator:
            input_data, labels = [x for x in batch]
            
            fs_optimizer.zero_grad()
            shaps = self.explainer(input_data)
            samp = S.sample(input_data.shape[0]*1, True)
            total = (shaps * samp).sum(dim=1)
            grand = imputer(input_data, samp)
            null = imputer(input_data, 0)
            fs_loss = loss_fn(total, grand - null)
            fs_loss.backward()
            fs_optimizer.step()

            mlp_optimizer.zero_grad()
            output = self(input_data)
            mlp_loss = self.lambda_*criterion_fair(self.explainer, torch.sigmoid(output), input_data, prot_col) + criterion_perf(output, labels)
            if fair_loss == 'sp':
                mlp_loss += criterion_standard_fair(torch.sigmoid(output), input_data, prot_col)
            elif fair_loss == 'eo':
                mlp_loss += criterion_standard_fair(torch.sigmoid(output), input_data, labels, prot_col)
            mlp_loss.backward()
            mlp_optimizer.step()
            torch.cuda.empty_cache()

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        dataloader = TensorDataset(X)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=False) 
        self.eval()
        preds = np.array([])
        with torch.no_grad():
            for inputs in dataloader:
                batch_preds = self._predict_batch(inputs[0])
                preds = np.append(preds, batch_preds)
        return (preds >=.5).astype(int)
    
    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        dataloader = TensorDataset(X)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=False) 
        self.eval()
        preds = np.array([])
        with torch.no_grad():
            for inputs in dataloader:
                batch_preds = self._predict_batch(inputs[0])
                preds = np.append(preds, batch_preds)
        return preds
    
    def _predict_batch(self, X):
        self.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        with torch.no_grad():
            logits = self(X)
            preds = torch.sigmoid(logits)
            preds = preds.detach().numpy().flatten()
            return preds
        
    def get_params(self, deep=False):
        return {'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'verbose': self.verbose,
                'seed': self.seed,
                'batch_size': self.batch_size,
                'lambda_': self.lambda_
               }
    
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=None, seed=None, batch_size=None, lambda_=None):
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if lr is not None:
            self.lr = lr
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if input_size is not None:
            self.input_size = input_size
        if output_size is not None:
            self.output_size = output_size
        if verbose is not None:
            self.verbose = verbose
        if seed is not None:
            self.seed = seed
        if batch_size is not None:
            self.batch_size = batch_size
        if lambda_ is not None:
            self.lambda_ = lambda_
        return self


if __name__ == '__main__':
    import FairDataLoader
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    import shap
    import pandas as pd
    X, y = FairDataLoader.get_diabetes_data()
    pc = 'gender'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)
    scl = MinMaxScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)
    mdl = EDMLP(X.shape[1], 50, 1, .001, 100, batch_size=128)
    mdl.fit(X_train, y_train, list(X.columns).index(pc), fair_loss= 'eo')
    preds = mdl.predict_proba(X_test)
    e = shap.Explainer(mdl.predict, pd.DataFrame(X_train, columns=X.columns).iloc[:100, :])
    shaps = e(X_test, silent=False)
    print(roc_auc_score(y_test, preds))
    print(f1_score(y_test, (preds>=.5).astype(int)))