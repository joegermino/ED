
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class EDLoss(nn.Module):
    def __init__(self):
        super(EDLoss, self).__init__()

    def forward(self, explainer, preds, X, prot_attr_idx):
        shaps = explainer(X)
        shaps = ((shaps - torch.min(shaps, axis=1).values.reshape(-1,1))/(torch.max(shaps, axis=1).values.reshape(-1,1)- torch.min(shaps, axis=1).values.reshape(-1,1))) # I'm going to hell for this line

        a0 = (X[:, prot_attr_idx] == 0)
        a1 = (X[:, prot_attr_idx] == 1) 

        if sum((preds.flatten()>=.5) & (a0)) == 0:
            s0_1_mean = torch.tensor(0)
        else:
            s0_1_mean = torch.mean(shaps[a0 & (preds.flatten()>=.5), :], axis=0)
        if sum((preds.flatten()>=.5) & (a1)) == 0:    
            s1_1_mean = torch.tensor(0)
        else:
            s1_1_mean = torch.mean(shaps[a1 & (preds.flatten()>=.5), :], axis=0)
        
        if sum((preds.flatten()<.5) & (a0)) == 0:
            s0_0_mean = torch.tensor(0)
        else:
            s0_0_mean = torch.mean(shaps[a0 & (preds.flatten()<.5), :], axis=0)
        if sum((preds.flatten()<.5) & (a1)) == 0:    
            s1_0_mean = torch.tensor(0)
        else:
            s1_0_mean = torch.mean(shaps[a1 & (preds.flatten()<.5), :], axis=0)
        
        return torch.sum(abs(s0_1_mean - s1_1_mean) + abs(s0_0_mean - s1_0_mean))
    