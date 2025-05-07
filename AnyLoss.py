import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

'''
Han, Doheon, Nuno Moniz, and Nitesh V. Chawla. 
"AnyLoss: Transforming Classification Metrics into Loss Functions." 
Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.
'''

class AnyLoss(nn.Module):
    def __init__(self, L):
        super(AnyLoss, self).__init__()
        self.L = L

    def forward(self, preds, trues):
        beta = 1
        y_pred = 1/(1+torch.exp(-self.L*(preds-0.5)))
        numerator = (1+beta**2)*torch.sum(trues*y_pred)
        denominator = (beta**2)*torch.sum(trues) + torch.sum(y_pred)
        return 1-(numerator/denominator)

