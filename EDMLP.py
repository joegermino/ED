import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from EDLoss import EDLoss
from SPLoss import SPLoss
from fastshap.utils import ShapleySampler
from AnyLoss import AnyLoss
import random

class EDMLP(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, early_stopping=False, patience=5):
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
        self.prot_col_idx = None
        self.patience = patience
        self.early_stopping = early_stopping

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

    def fit(self, X, y, prot_col_idx=None, sample_weight=None):
        '''
        X: 2-dimensional np.ndarray or torch.Tensor, input features minus the Sensitive Attribute
        y: 1-dimensional np.ndarray or torch.Tensor, true values
        '''
        if prot_col_idx is None and self.prot_col_idx is None:
            raise ValueError("prot_col_idx cannot be null")
        if prot_col_idx is not None:
            self.prot_col_idx = prot_col_idx
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)

        fs_optimizer = optim.Adam(self.explainer.parameters(), lr=self.lr)
        mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=self.lr)
        criterion_fair = EDLoss()

        loss_fn = nn.MSELoss()
        S = ShapleySampler(X.shape[1])
        best_loss = np.inf
        patience = 0
        for i in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            l = self._train_epoch(dataloader, criterion_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx)
            if l < best_loss*.99:
                best_loss = l
                patience = 0
            else:
                patience += 1
            if patience == self.patience and self.early_stopping:
                print(f"Early Stopping at Epoch {i}")
                break
        
    def _train_epoch(self, iterator, criterion_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx):
        self.train()
        imputer = lambda X, s: torch.sigmoid(self(X*s))
        epoch_loss = 0
        for batch in iterator:
            input_data, _ = [x for x in batch]
            
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
            mlp_loss = criterion_fair(self.explainer, input_data, prot_col_idx)
            mlp_loss.backward()
            mlp_optimizer.step()
            torch.cuda.empty_cache()
            epoch_loss += mlp_loss.item()
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
        
    def get_params(self, deep=False):
        return {'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'verbose': self.verbose,
                'seed': self.seed,
                'batch_size': self.batch_size,
                'patience': self.patience,
                'early_stopping': self.early_stopping
               }
    
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=None, seed=None, batch_size=None, patience=None, early_stopping=None):
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
        if patience is not None:
            self.patience = patience
        if early_stopping is not None:
            self.early_stopping = early_stopping
        return self
    
class EDMLP_F1(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, L=75, early_stopping=False, patience=5):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        self.batch_size = batch_size
        torch.manual_seed(self.seed)
        super(EDMLP_F1, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)   
        )
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.L = L
        self.prot_col_idx = None
        self.patience = patience
        self.early_stopping = early_stopping

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
    
    def fit(self, X, y, prot_col_idx=None, sample_weight=None):
        if prot_col_idx is None and self.prot_col_idx is None:
            raise ValueError("prot_col_idx cannot be null")
        if prot_col_idx is not None:
            self.prot_col_idx = prot_col_idx
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)

        fs_optimizer = optim.Adam(self.explainer.parameters(), lr=self.lr)
        mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=self.lr)
        criterion_fair = EDLoss()
        criterion_perf = AnyLoss(self.L)
        loss_fn = nn.MSELoss()
        S = ShapleySampler(X.shape[1])
        best_loss = np.inf
        patience = 0
        for i in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            l = self._train_epoch(dataloader, criterion_fair, criterion_perf, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx)
            if l < best_loss*.99:
                best_loss = l
                patience = 0
            else:
                patience += 1
            if patience == self.patience and self.early_stopping:
                print(f"Early Stopping at Epoch {i}")
                break
        
    def _train_epoch(self, iterator, criterion_fair, criterion_perf, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx):
        self.train()
        imputer = lambda X, s: torch.sigmoid(self(X*s))
        epoch_loss = 0
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
            mlp_loss = criterion_fair(self.explainer, input_data, prot_col_idx) + criterion_perf(torch.sigmoid(output), labels)
            mlp_loss.backward()
            mlp_optimizer.step()
            torch.cuda.empty_cache()
            epoch_loss += mlp_loss.item()
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
        
    def get_params(self, deep=False):
        return {'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'verbose': self.verbose,
                'seed': self.seed,
                'batch_size': self.batch_size,
                'L': self.L,
                'patience': self.patience,
                'early_stopping': self.early_stopping
               }
    
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=None, seed=None, batch_size=None, L=None, patience=None, early_stopping=None):
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
        if L is not None:
            self.L = L
        if patience is not None:
            self.patience = patience
        if early_stopping is not None:
            self.early_stopping = early_stopping
        return self

class EDMLP_F1_SP(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, L=75, early_stopping=False, patience=5):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        self.batch_size = batch_size
        torch.manual_seed(self.seed)
        random.seed(seed)
        super(EDMLP_F1_SP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
             nn.Linear(hidden_size, output_size)   
        )
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.L = L
        self.prot_col_idx = None
        self.fs_loss = []
        self.ed_loss = []
        self.sp_loss = []
        self.f1_loss = []
        self.patience = patience
        self.early_stopping = early_stopping

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
    
    def fit(self, X, y, prot_col_idx=None, sample_weight=None):
        if prot_col_idx is None and self.prot_col_idx is None:
            raise ValueError("prot_col_idx cannot be null")
        if prot_col_idx is not None:
            self.prot_col_idx = prot_col_idx
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)

        fs_optimizer = optim.Adam(self.explainer.parameters(), lr=self.lr)
        mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=self.lr)
        criterion_fair = EDLoss()
        criterion_perf = AnyLoss(self.L)
        criterion_standard_fair = SPLoss()
        loss_fn = nn.MSELoss()
        S = ShapleySampler(X.shape[1])
        best_loss = np.inf
        patience = 0
        for i in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            l = self._train_epoch(dataloader, criterion_fair, criterion_perf, criterion_standard_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx)
            if l < best_loss*.99:
                best_loss = l
                patience = 0
            else:
                patience += 1
            if patience == self.patience and self.early_stopping:
                print(f"Early Stopping at Epoch {i}")
                break
        
    def _train_epoch(self, iterator, criterion_fair, criterion_perf, criterion_standard_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx):
        self.train()
        imputer = lambda X, s: torch.sigmoid(self(X*s))
        running_fs_loss = 0
        running_f1_loss = 0
        running_sp_loss = 0
        running_ed_loss = 0
        epoch_loss = 0
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
            running_fs_loss += fs_loss.item()

            mlp_optimizer.zero_grad()
            output = self(input_data)
            ed_loss = criterion_fair(self.explainer, input_data, prot_col_idx)
            f1_loss = criterion_perf(torch.sigmoid(output), labels)
            sp_loss = criterion_standard_fair(torch.sigmoid(output), input_data[:, prot_col_idx])
            mlp_loss =  ed_loss + f1_loss + sp_loss

            mlp_loss.backward()
            mlp_optimizer.step()
            running_f1_loss += f1_loss.item()
            running_ed_loss += ed_loss.item()
            running_sp_loss += sp_loss.item()
            epoch_loss += mlp_loss.item()

            torch.cuda.empty_cache()
        self.fs_loss.append(running_fs_loss)
        self.ed_loss.append(running_ed_loss)
        self.sp_loss.append(running_sp_loss)
        self.f1_loss.append(running_f1_loss)
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
        
    def get_params(self, deep=False):
        return {'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'verbose': self.verbose,
                'seed': self.seed,
                'batch_size': self.batch_size,
                'L': self.L,
                'patience': self.patience,
                'early_stopping': self.early_stopping
               }
    
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=None, seed=None, batch_size=None, L=None, patience=None, early_stopping=None):
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
        if L is not None:
            self.L = L
        if patience is not None:
            self.patience = patience
        if early_stopping is not None:
            self.early_stopping = early_stopping
        return self
    
class EDMLP_SP(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, early_stopping=False, patience=5):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        self.batch_size = batch_size
        torch.manual_seed(self.seed)
        super(EDMLP_SP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
             nn.Linear(hidden_size, output_size)   
        )
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.prot_col_idx = None
        self.fs_loss = []
        self.ed_loss = []
        self.sp_loss = []
        self.patience = patience
        self.early_stopping = early_stopping

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
    
    def fit(self, X, y, prot_col_idx=None, sample_weight=None):
        if prot_col_idx is None and self.prot_col_idx is None:
            raise ValueError("prot_col_idx cannot be null")
        if prot_col_idx is not None:
            self.prot_col_idx = prot_col_idx
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)

        fs_optimizer = optim.Adam(self.explainer.parameters(), lr=self.lr)
        mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=self.lr)
        criterion_fair = EDLoss()
        criterion_standard_fair = SPLoss()
        loss_fn = nn.MSELoss()
        S = ShapleySampler(X.shape[1])
        best_loss = np.inf
        patience = 0
        for i in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            l = self._train_epoch(dataloader, criterion_fair, criterion_standard_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx)
            if l < best_loss*.99:
                best_loss = l
                patience = 0
            else:
                patience += 1
            if patience == self.patience and self.early_stopping:
                print(f"Early Stopping at Epoch {i}")
                break
        
    def _train_epoch(self, iterator, criterion_fair, criterion_standard_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx):
        self.train()
        imputer = lambda X, s: torch.sigmoid(self(X*s))
        running_fs_loss = 0
        running_sp_loss = 0
        running_ed_loss = 0
        epoch_loss = 0
        for batch in iterator:
            input_data, _ = [x for x in batch]
            
            fs_optimizer.zero_grad()
            shaps = self.explainer(input_data)
            samp = S.sample(input_data.shape[0]*1, True)
            total = (shaps * samp).sum(dim=1)
            grand = imputer(input_data, samp)
            null = imputer(input_data, 0)
            fs_loss = loss_fn(total, grand - null)
            fs_loss.backward()
            fs_optimizer.step()
            running_fs_loss += fs_loss.item()

            mlp_optimizer.zero_grad()
            output = self(input_data)
            ed_loss = criterion_fair(self.explainer, input_data, prot_col_idx)
            sp_loss = criterion_standard_fair(torch.sigmoid(output), input_data[:, prot_col_idx])
            mlp_loss =  ed_loss + sp_loss

            mlp_loss.backward()
            mlp_optimizer.step()
            running_ed_loss += ed_loss.item()
            running_sp_loss += sp_loss.item()
            epoch_loss += mlp_loss.item()

            torch.cuda.empty_cache()
        self.fs_loss.append(running_fs_loss)
        self.ed_loss.append(running_ed_loss)
        self.sp_loss.append(running_sp_loss)
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
        
    def get_params(self, deep=False):
        return {'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'verbose': self.verbose,
                'seed': self.seed,
                'batch_size': self.batch_size,
                'patience': self.patience,
                'early_stopping': self.early_stopping
               }
    
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=None, seed=None, batch_size=None, L=None, patience=None, early_stopping=None):
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
        if patience is not None:
            self.patience = patience
        if early_stopping is not None:
            self.early_stopping = early_stopping
        return self

class EDMLP_CURRICULUM(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1, lr=.001, num_epochs=100, verbose=True, seed=12345, batch_size=32, L=75, early_stopping=False, patience=5):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        self.batch_size = batch_size
        torch.manual_seed(self.seed)
        super(EDMLP_CURRICULUM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
             nn.Linear(hidden_size, output_size)   
        )
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.L = L
        self.prot_col_idx = None
        self.fs_loss = []
        self.ed_loss = []
        self.sp_loss = []
        self.f1_loss = []
        self.patience = patience
        self.early_stopping = early_stopping

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
    
    def fit(self, X, y, prot_col_idx=None, sample_weight=None):
        if prot_col_idx is None and self.prot_col_idx is None:
            raise ValueError("prot_col_idx cannot be null")
        if prot_col_idx is not None:
            self.prot_col_idx = prot_col_idx
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=self.batch_size, shuffle=True)

        fs_optimizer = optim.Adam(self.explainer.parameters(), lr=self.lr)
        mlp_optimizer = optim.Adam(self.mlp.parameters(), lr=self.lr)
        criterion_fair = EDLoss()
        criterion_perf = AnyLoss(self.L)
        criterion_standard_fair = SPLoss()
        loss_fn = nn.MSELoss()
        S = ShapleySampler(X.shape[1])
        best_loss = np.inf
        patience = 0
        for i in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            if (i//(self.num_epochs/4)+1) == 1:
                sample_size = .05
            elif (i//(self.num_epochs/4)+1) == 2:
                sample_size = .10
            elif (i//(self.num_epochs/4)+1) == 3:
                sample_size = .20
            elif (i//(self.num_epochs/4)+1) == 4:
                sample_size = .50
            l = self._train_epoch(dataloader, criterion_fair, criterion_perf, criterion_standard_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx, sample_size)
            if l < best_loss*.99:
                best_loss = l
                patience = 0
            else:
                patience += 1
            if patience == self.patience and self.early_stopping:
                print(f"Early Stopping at Epoch {i}")
                break
        
    def _train_epoch(self, iterator, criterion_fair, criterion_perf, criterion_standard_fair, fs_optimizer, mlp_optimizer, loss_fn, S, prot_col_idx, sample_size):
        self.train()
        imputer = lambda X, s: torch.sigmoid(self(X*s))
        running_fs_loss = 0
        running_f1_loss = 0
        running_sp_loss = 0
        running_ed_loss = 0
        epoch_loss = 0
        for batch in iterator:
            input_data, labels = [x for x in batch]
            sample_idx = random.sample(range(len(input_data)), max(int(sample_size*len(input_data)), 2)) # Sample size has to be greater than 1 or FastSHAP breaks
            sampled_input = input_data[sample_idx]
            fs_optimizer.zero_grad()
            shaps = self.explainer(sampled_input)
            samp = S.sample(sampled_input.shape[0]*1, True)
            total = (shaps * samp).sum(dim=1)
            grand = imputer(sampled_input, samp)
            null = imputer(sampled_input, 0)
            fs_loss = loss_fn(total, grand - null)
            fs_loss.backward()
            fs_optimizer.step()
            running_fs_loss += fs_loss.item()

            mlp_optimizer.zero_grad()
            output = self(input_data)
            ed_loss = criterion_fair(self.explainer, sampled_input, prot_col_idx)
            f1_loss = criterion_perf(torch.sigmoid(output), labels)
            sp_loss = criterion_standard_fair(torch.sigmoid(output), input_data[:, prot_col_idx])
            mlp_loss =  len(input_data) / len(sample_idx) * ed_loss + f1_loss + sp_loss

            mlp_loss.backward()
            mlp_optimizer.step()
            running_f1_loss += f1_loss.item()
            running_ed_loss += ed_loss.item()
            running_sp_loss += sp_loss.item()
            epoch_loss += mlp_loss.item()

            torch.cuda.empty_cache()
        self.fs_loss.append(running_fs_loss)
        self.ed_loss.append(running_ed_loss)
        self.sp_loss.append(running_sp_loss)
        self.f1_loss.append(running_f1_loss)
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
        
    def get_params(self, deep=False):
        return {'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'verbose': self.verbose,
                'seed': self.seed,
                'batch_size': self.batch_size,
                'L': self.L,
                'patience': self.patience,
                'early_stopping': self.early_stopping
               }
    
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, input_size=None, output_size=None, verbose=None, seed=None, batch_size=None, L=None, patience=None, early_stopping=None):
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
        if L is not None:
            self.L = L
        if patience is not None:
            self.patience = patience
        if early_stopping is not None:
            self.early_stopping = early_stopping
        return self

if __name__ == '__main__':
    import FairDataLoader
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    import pandas as pd

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
    # X, y = FairDataLoader.get_adult_data()
    # pc = 'sex__Male'
    # X, y = FairDataLoader.get_diabetes_data()
    # pc = 'gender'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)
    # X_train = X_train.drop(pc, axis=1)
    # X_test = X_test.drop(pc, axis=1)
    scl = MinMaxScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)
    mdl = EDMLP_CURRICULUM(X_train.shape[1], 50, 1, .001, 200, batch_size=256, early_stopping=False)
    # mdl.fit(X_train, y_train, A_train)
    mdl.fit(X_train, y_train, list(X.columns).index(pc))
    preds = mdl.predict_proba(X_test)
    # e = fastshap(X_train, mdl, 50, 1234)
    print(roc_auc_score(y_test, preds))
    print(f1_score(y_test, (preds>=.5).astype(int)))