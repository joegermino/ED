import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class EOLoss(nn.Module):
    def __init__(self):
        super(EOLoss, self).__init__()

    def forward(self, preds, X, y, prot_attr_idx):
        pos1 = preds[(X[:, prot_attr_idx]==1) & (y==1).flatten()].sum()/preds[(X[:, prot_attr_idx]==1) & (y==1).flatten()].shape[0]
        neg1 = preds[((X[:, prot_attr_idx]==0)) & (y==1).flatten()].sum()/preds[((X[:, prot_attr_idx]==0)) & (y==1).flatten()].shape[0]
        pos0 = preds[(X[:, prot_attr_idx]==1) & (y==0).flatten()].sum()/preds[(X[:, prot_attr_idx]==1) & (y==0).flatten()].shape[0]
        neg0 = preds[((X[:, prot_attr_idx]==0)) & (y==0).flatten()].sum()/preds[((X[:, prot_attr_idx]==0)) & (y==0).flatten()].shape[0]
        eo = abs(pos1 - neg1) + abs(pos0 - neg0)
        return eo