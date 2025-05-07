import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from AnyLoss import AnyLoss
from SPLoss import SPLoss

class MLP_F1(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, L=75, early_stopping=False, patience=5):
        super(MLP_F1, self).__init__()
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
        self.L = L
        self.patience = patience
        self.early_stopping = early_stopping

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
                'batch_size': self.batch_size,
                'L': self.L
               }
    
    def fit(self, X, y, sample_weight=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)
        # if sample_weight is not None:
        #     if type(sample_weight) != torch.Tensor:
        #         sample_weight = torch.tensor(np.array(sample_weight))
        #     criterion = nn.BCEWithLogitsLoss(weight=sample_weight)
        # else:
        #     criterion = nn.BCEWithLogitsLoss()
        criterion = AnyLoss(L=self.L)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        best_loss = np.inf
        patience = 0
        for i in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            l = self._train_epoch(dataloader, optimizer, criterion)
            if l < best_loss*.99:
                best_loss = l
                patience = 0
            else:
                patience += 1
            if patience == self.patience and self.early_stopping:
                print(f"Early Stopping at Epoch {i}")
                break
    
    def _train_epoch(self, iterator, optimizer, criterion):
        self.train()
        epoch_loss = 0
        for batch in iterator:
            optimizer.zero_grad()
            input_data, labels = [x for x in batch]
            output = torch.sigmoid(self(input_data))
            loss = criterion(output.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
        return epoch_loss/len(iterator)
            
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
        
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=False, batch_size=None, L=None):
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
        if L is not None:
            self.L = L
        return self

class MLP_SP(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, L=75, early_stopping=False, patience=5):
        super(MLP_SP, self).__init__()
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
        self.L = L
        self.patience = patience
        self.early_stopping = early_stopping
        self.prot_col_idx = None

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
                'batch_size': self.batch_size,
                'L': self.L
               }
    
    def fit(self, X, y, prot_col_idx=None, sample_weight=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        if prot_col_idx is None and self.prot_col_idx is None:
            raise ValueError("prot_col_idx cannot be null")
        if prot_col_idx is not None:
            self.prot_col_idx = prot_col_idx
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)
        criterion = SPLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        best_loss = np.inf
        patience = 0
        for i in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            l = self._train_epoch(dataloader, optimizer, criterion, prot_col_idx)
            if l < best_loss*.99:
                best_loss = l
                patience = 0
            else:
                patience += 1
            if patience == self.patience and self.early_stopping:
                print(f"Early Stopping at Epoch {i}")
                break
    
    def _train_epoch(self, iterator, optimizer, criterion, prot_col_idx):
        self.train()
        epoch_loss = 0
        for batch in iterator:
            optimizer.zero_grad()
            input_data, labels = [x for x in batch]
            output = torch.sigmoid(self(input_data))
            loss = criterion(torch.sigmoid(output), input_data[:, prot_col_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
        return epoch_loss/len(iterator)
            
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
        
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=False, batch_size=None, L=None):
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
        if L is not None:
            self.L = L
        return self

class MLP_F1_SP(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, L=75, early_stopping=False, patience=5):
        super(MLP_F1_SP, self).__init__()
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
        self.L = L
        self.patience = patience
        self.early_stopping = early_stopping
        self.prot_col_idx = None

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
                'batch_size': self.batch_size,
                'L': self.L
               }
    
    def fit(self, X, y, prot_col_idx=None, sample_weight=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        if prot_col_idx is None and self.prot_col_idx is None:
            raise ValueError("prot_col_idx cannot be null")
        if prot_col_idx is not None:
            self.prot_col_idx = prot_col_idx
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)
        # if sample_weight is not None:
        #     if type(sample_weight) != torch.Tensor:
        #         sample_weight = torch.tensor(np.array(sample_weight))
        #     criterion = nn.BCEWithLogitsLoss(weight=sample_weight)
        # else:
        #     criterion = nn.BCEWithLogitsLoss()
        criterion = AnyLoss(L=self.L)
        criterion_fair = SPLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        best_loss = np.inf
        patience = 0
        for i in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            l = self._train_epoch(dataloader, optimizer, criterion, criterion_fair, prot_col_idx)
            if l < best_loss*.99:
                best_loss = l
                patience = 0
            else:
                patience += 1
            if patience == self.patience and self.early_stopping:
                print(f"Early Stopping at Epoch {i}")
                break
    
    def _train_epoch(self, iterator, optimizer, criterion, criterion_fair, prot_col_idx):
        self.train()
        epoch_loss = 0
        for batch in iterator:
            optimizer.zero_grad()
            input_data, labels = [x for x in batch]
            output = torch.sigmoid(self(input_data))
            loss = criterion(output.squeeze(), labels.squeeze()) + criterion_fair(torch.sigmoid(output), input_data[:, prot_col_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
        return epoch_loss/len(iterator)
            
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
        
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=False, batch_size=None, L=None):
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
        if L is not None:
            self.L = L
        return self


if __name__ == '__main__':
    import FairDataLoader
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    import pandas as pd
    from fastshap.utils import ShapleySampler

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
    
    X, y = FairDataLoader.get_dutch_census_data()
    pc = 'sex_2'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)
    scl = MinMaxScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)
    mdl = MLP_F1(X.shape[1], 100, 1, .001, 100, batch_size=512, L=5, early_stopping=True, seed=420)
    mdl.fit(X_train, y_train)
    preds = mdl.predict_proba(X_test)
    print(roc_auc_score(y_test, preds))
    print(f1_score(y_test, (preds>=.5).astype(int)))
    # e = fastshap(X_train, mdl, 50, 1234)
    