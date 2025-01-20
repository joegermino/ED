
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, preds, X, prot_attr_idx):
        sp = preds[X[:, prot_attr_idx]==1].sum()/preds[X[:, prot_attr_idx]==1].shape[0] - preds[~(X[:, prot_attr_idx]==1)].sum()/preds[~(X[:, prot_attr_idx]==1)].shape[0]
        return abs(sp)
    