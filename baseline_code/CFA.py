import numpy as np
import torch

from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.linear_model import LassoLars
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import dropout_adj, convert
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import random

class CFA(nn.Module):
    def __init__(self, input_size, hidden_size=10, lr=.001, num_epochs=100, dropout=0.1, top_k=1, weight_decay=.00001, lambda_=1, opt_start_epoch=400, top_ratio=.2, seed=12345, verbose=True):
        super(CFA, self).__init__()
        self.layers = []
        self.activation = nn.LeakyReLU()
        self.num_epochs = num_epochs
        self.top_k = top_k
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_ = lambda_
        self.opt_start_epoch = opt_start_epoch
        self.top_ratio = top_ratio
        self.dropout = dropout
        self.seed = seed
        self.verbose = verbose
        self.hidden_size = hidden_size
        self.input_size = input_size
        torch.manual_seed(self.seed)

        self.layers.append(nn.Linear(input_size, hidden_size, bias=False))
        self.layers.append(self.activation)
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_size, 2, bias=False))
        self.layers = nn.ModuleList(self.layers)

    def get_params(self, deep=False):
        return {'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'dropout': self.dropout,
                'top_k': self.top_k,
                'weight_decay': self.weight_decay,
                'lambda_': self.lambda_,
                'opt_start_epoch': self.opt_start_epoch,
                'top_ratio': self.top_ratio,
                'seed': self.seed,
                'input_size': self.input_size,
                'verbose': self.verbose
               }
    
    def set_params(self, hidden_size=None, lr=None, num_epochs=None, dropout=None, top_k=None, weight_decay=None, lambda_=None, opt_start_epoch=None, top_ratio=None, seed=None, input_size=None, verbose=None):
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if lr is not None:
            self.lr = lr
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if dropout is not None:
            self.dropout = dropout
        if top_k is not None:
            self.top_k = top_k
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if lambda_ is not None:
            self.lambda_ = lambda_
        if opt_start_epoch is not None:
            self.opt_start_epoch = opt_start_epoch
        if top_ratio is not None:
            self.top_ratio = top_ratio
        if seed is not None:
            self.seed = seed
        if input_size is not None:
            self.input_size = input_size
        if verbose is not None:
            self.verbose = verbose
        return self

    def build_relationship(self, x, thresh=0.25):
        df_euclid = 1 / (1 + distance_matrix(x.T.T, x.T.T))
        idx_map = []
        for ind in range(df_euclid.shape[0]):
            max_sim = np.sort(df_euclid[ind, :])[-2]
            neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
            random.seed(self.seed)
            random.shuffle(neig_id)
            for neig in neig_id:
                if neig != ind:
                    idx_map.append([ind, neig])
        idx_map =  np.array(idx_map)

        return idx_map

    def build_edge_index(self, X, y):
        edges_unordered = self.build_relationship(X, thresh=0.8)
        features = sp.csr_matrix(X, dtype=np.float32)
        labels = y

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        return edge_index

    def fit(self, X, y, prot_col_idx):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        dataloader = TensorDataset(X, y)
        dataloader = DataLoader(dataloader, batch_size=X.shape[0], shuffle=True)

        sens = X[:, prot_col_idx]
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        edge_index = self.build_edge_index(X, y)
        interpretation = Interpreter(features=X, edge_index=edge_index,\
            utility_labels=y, sensitive_labels=sens, top_ratio=self.top_ratio, \
            topK=self.top_k)
        
        for i in tqdm(range(self.num_epochs), disable=not self.verbose):
            self._train_epoch(dataloader, i, optimizer, interpretation, sens)

    def _train_epoch(self, dataloader, epoch, optimizer, interpretation, sens):
        self.train()
        optimizer.zero_grad()
        X = dataloader.dataset.tensors[0]
        y = dataloader.dataset.tensors[1].squeeze()

        hidden, output = self(x=X)
        l_classification = F.cross_entropy(output, y.long())
        
        if epoch < self.opt_start_epoch: # warm start
            loss_train = l_classification
        else: # apply distance-based loss
            X_masked, _ = interpretation.generate_masked_X(idx=X.index, model=self)
            hidden_masked, output_masked = self(x=X_masked)
            l_distance = self.group_distance(hidden, sens.numpy(), \
                y.numpy())
            l_distance_masked = self.group_distance(hidden_masked, sens.numpy(), \
                y.numpy())
            loss_train = l_classification + self.lambda_*(l_distance+l_distance_masked)
        loss_train.backward()
        optimizer.step()

    def forward(self, x):
        hidden_emb = self.layers[0](x)
        h = hidden_emb
        for layer in self.layers[1:]:
            h = layer(h)
        return hidden_emb, h
    
    def predict_proba(self, X):
        return self.predict_proba_1(X).detach().numpy()[:, 1]

    def predict_proba_1(self, x):
        x = torch.tensor(x).float()
        h = x
        for layer in self.layers:
            h = layer(h)
        h = nn.Softmax(dim=1)(h) # transform to probability
        return h
    
    def predict(self, x):
        x = torch.tensor(x).float()
        h = x
        for layer in self.layers:
            h = layer(h)
        h = nn.Softmax(dim=1)(h) # transform to probability
        return (h.detach().numpy()>=.5).astype(int)

    def group_distance(self, hidden, sens, labels):
        # return the embedding distance of two subgroups
        sens_ = sens
        labels_ = labels
        hidden_ = hidden

        # obtain idx (idx_s0_y1) for data whose sensitive label=0 and utility label=1
        idx_s0 = sens_==0
        idx_s1 = sens_==1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels_==1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels_==1)
        idx_s0_y0 = np.bitwise_and(idx_s0, labels_==0)
        idx_s1_y0 = np.bitwise_and(idx_s1, labels_==0)

        hidden_s0_y1 = hidden_[idx_s0_y1]
        hidden_s1_y1 = hidden_[idx_s1_y1]
        hidden_s0_y0 = hidden_[idx_s0_y0]
        hidden_s1_y0 = hidden_[idx_s1_y0]

        sample_number = int(min(hidden_s0_y1.shape[0], hidden_s1_y1.shape[0])/2)
        random_idx_s0_y1 = random.sample(range(hidden_s0_y1.shape[0]), sample_number)
        random_idx_s1_y1 = random.sample(range(hidden_s1_y1.shape[0]), sample_number)

        sample_number = int(min(hidden_s0_y0.shape[0], hidden_s1_y0.shape[0])/2)
        random_idx_s0_y0 = random.sample(range(hidden_s0_y0.shape[0]), sample_number)
        random_idx_s1_y0 = random.sample(range(hidden_s1_y0.shape[0]), sample_number)

        x1 = hidden_s0_y1[random_idx_s0_y1]
        y1 = hidden_s1_y1[random_idx_s1_y1]
        x0 = hidden_s0_y0[random_idx_s0_y0]
        y0 = hidden_s1_y0[random_idx_s1_y0]

        distance_1 = self.SWloss_fast(x1, y1)
        distance_0 = self.SWloss_fast(x0, y0)
        
        return distance_1+distance_0
    
    def SWloss_fast(self, x, y, L=512):
        r = torch.randn(x.shape[1], L)
        xt = x @ r # (N, 8)
        yt = y @ r # (N, 8)
        xt = torch.sort(xt, 0)[0]
        yt = torch.sort(yt, 0)[0]
        t = xt - yt # (N, 3)
        SW = (t**2).sum()/L
        return SW


class GraphLIME_speedup:
    def __init__(self, X, edge_index, hop=1, rho=0.1):
        self.hop = hop
        self.rho = rho
        self.subset_dict = dict()
        num_nodes = X.shape[0]
        self.X = X
        self.edge_index = edge_index

        for node_idx in range(X.shape[0]):
            subset, edge_index_new, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.hop, edge_index, relabel_nodes=True,
                num_nodes=num_nodes, flow="source_to_target")
            self.subset_dict[node_idx] = subset

    def __subgraph__(self, node_idx, y, **kwargs):
        subset = self.subset_dict[node_idx]
        x = self.X[subset]
        y = y[subset]
        return x, y

    def __init_predict__(self, model, **kwargs):
        model.eval()
        with torch.no_grad():
            _, logits = model(x=self.X, **kwargs)
            probas = nn.Softmax(dim=1)(logits) # transform to probability
        return probas
    
    def __compute_kernel__(self, x, reduce):
        assert x.ndim == 2, x.shape
        n, d = x.shape
        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2
        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)
        std = np.sqrt(d)  
        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))  # (n, n, 1) or (n, n, d)
        return K
    
    def __compute_gram_matrix__(self, x):

        # more stable and accurate implementation
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)

        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

        return G
        
    def explain_node(self, node_idx, x, edge_index, **kwargs):
        probas = self.__init_predict__(x, edge_index, **kwargs)

        x, probas, _, _, _, _ = self.__subgraph__(
            node_idx, x, probas, edge_index, **kwargs)

        x = x.detach().numpy()  # (n, d)
        y = probas.detach().numpy()  # (n, classes)

        n, d = x.shape

        K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
        L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

        K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
        L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

        K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
        L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)

        solver = LassoLars(self.rho, fit_intercept=False, normalize=False, positive=True)

        solver.fit(K_bar * n, L_bar * n)

        return solver.coef_

    def explain_nodes(self, model, node_idxs, **kwargs):
        all_probas = self.__init_predict__(model, **kwargs)

        coefs = []
        for i in range(len(node_idxs)):
            node_idx = node_idxs[i]

            subset = self.subset_dict[node_idx]
            x = self.X[subset].detach().numpy()
            y = all_probas[subset].detach().numpy()
            
            n, d = x.shape

            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)

            solver = LassoLars(self.rho, fit_intercept=False, normalize=False, positive=True)

            solver.fit(K_bar * n, L_bar * n)

            coef = solver.coef_

            coefs.append(coef)
        return abs(np.array(coefs))

class Interpreter:
    def __init__(self, features, edge_index, utility_labels, sensitive_labels, top_ratio=0.2, \
        topK=1, rho=0.1):
        self.explainer = GraphLIME_speedup(X=features, edge_index=edge_index, hop=1, rho=rho)
        self.X = features
        self.edge_index = edge_index
        self.utility_labels = utility_labels
        self.sensitive_labels = sensitive_labels
        self.K = topK
        self.top_ratio = top_ratio # select the top high fidelity users

    def generate_masked_X(self, idx, model):
        X_masked = self.X[idx].detach().clone()
        # remove top K important features
        coefs = torch.tensor(self.explainer.explain_nodes(model, idx.tolist()))
        indices = coefs.argsort()[:, -self.K:]
        # use scatter to set the values
        if self.K != 0:
            X_masked = X_masked.scatter(1, indices, torch.zeros((len(idx),self.K)))
        return X_masked, coefs.tolist()

    def interprete(self, model, idx):
        model.eval()

        # find the index of subgroups
        sens = self.sensitive_labels[idx]
        idx_s0 = (sens == 0).nonzero().squeeze().tolist()
        idx_s1 = (sens == 1).nonzero().squeeze().tolist()

        # compute fidelity_prob for each user
        output = model.predict_proba(x=self.X[idx]) # N, 2
        preds = (output.argmax(axis=1)).type_as(self.utility_labels) # N
        preds_prob = output.gather(1, preds.view(-1,1))

        X_masked, coefs = self.generate_masked_X(idx, model) # idx, F
        masked_output = model.predict_proba(x=X_masked)
        masked_preds = (masked_output.argmax(axis=1)).type_as(self.utility_labels)
        masked_preds_prob = masked_output.gather(1, masked_preds.view(-1,1))
        
        F = preds_prob - masked_preds_prob # N, 1

        # find the top high fidelity
        K = int(self.top_ratio*len(idx))
        value, max_fidelity_index = F.topk(k=K, dim=0)
        max_fidelity_index_set = set(max_fidelity_index.squeeze().tolist())
        group0 = set(idx_s0).intersection(max_fidelity_index_set)
        group1 = set(idx_s1).intersection(max_fidelity_index_set)

        # REF
        p0 = float(len(group0))/float(len(idx_s0))
        p1 = float(len(group1))/float(len(idx_s1))

        # VEF
        k0 = int(self.top_ratio*len(idx_s0))
        k1 = int(self.top_ratio*len(idx_s1))
        _, max_fidelity_index0 = F[idx_s0].topk(k=k0, dim=0)
        _, max_fidelity_index1 = F[idx_s1].topk(k=k1, dim=0)
        max_fidelity_index0 = max_fidelity_index0.squeeze().tolist()
        max_fidelity_index1 = max_fidelity_index1.squeeze().tolist()
        original_acc_0 = accuracy_score(self.utility_labels[idx][max_fidelity_index0].numpy(), preds[max_fidelity_index0].numpy())
        masked_acc_0 = accuracy_score(self.utility_labels[idx][max_fidelity_index0].numpy(), masked_preds[max_fidelity_index0].numpy())
        original_acc_1 = accuracy_score(self.utility_labels[idx][max_fidelity_index1].numpy(), preds[max_fidelity_index1].numpy())
        masked_acc_1 = accuracy_score(self.utility_labels[idx][max_fidelity_index1].numpy(), masked_preds[max_fidelity_index1].numpy())
        top_acc_fidelity_g0 = original_acc_0 - masked_acc_0
        top_acc_fidelity_g1 = original_acc_1 - masked_acc_1

        return p0, p1, abs(p0-p1), \
        top_acc_fidelity_g0, top_acc_fidelity_g1, abs(top_acc_fidelity_g0-top_acc_fidelity_g1)


if __name__ == '__main__':
    import FairDataLoader
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    X, y = FairDataLoader.get_adult_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)
    scl = MinMaxScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)
    mdl = CFA(X.shape[1], 50, .001, 10)
    mdl.fit(X_train, y_train, list(X.columns).index('sex__Male'))
    preds = mdl.predict_proba_1(X_test).detach().numpy()[:, 1]
    print(roc_auc_score(y_test, preds))