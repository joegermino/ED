import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class EOLoss(nn.Module):
    def __init__(self):
        super(EOLoss, self).__init__()

    def forward(self, preds, X, y, sas):
        pos1 = preds[(sas == 1).flatten() & (y==1).flatten()].sum()/preds[(sas == 1).flatten() & (y==1).flatten()].shape[0]
        neg1 = preds[(sas == 0).flatten() & (y==1).flatten()].sum()/preds[(sas == 0).flatten() & (y==1).flatten()].shape[0]
        pos0 = preds[(sas == 1).flatten() & (y==0).flatten()].sum()/preds[(sas == 1).flatten() & (y==0).flatten()].shape[0]
        neg0 = preds[(sas == 0).flatten() & (y==0).flatten()].sum()/preds[(sas == 0).flatten() & (y==0).flatten()].shape[0]
        eo = abs(pos1 - neg1) + abs(pos0 - neg0)
        return eo