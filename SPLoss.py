import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, preds, sas):
        sp = preds[(sas == 1).flatten()].sum()/preds[(sas == 1).flatten()].shape[0] - preds[(sas == 0).flatten()].sum()/preds[(sas == 0).flatten()].shape[0]
        return abs(sp)