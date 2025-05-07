import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class EDLoss(nn.Module):
    def __init__(self):
        super(EDLoss, self).__init__()

    def forward(self, explainer, X, prot_col_idx):
        X_copy = X.clone()
        X_copy[:, prot_col_idx].apply_(lambda x: 1 if x==0 else 0 if x==1 else torch.nan)
        #     x = torch.cat([X, X_copy]) # TODO: Make sure this is working properly
        shaps = explainer(X)
        shaps_counter = explainer(X_copy)

        mean_euclid_dist = torch.mean(abs(shaps-shaps_counter),axis=0)
        return mean_euclid_dist.sum()

        # X_copy = X.clone()
        # X_copy[:, prot_col_idx].apply_(lambda x: 1 if x==0 else 0 if x==1 else torch.nan)

        # x = torch.cat([X, X_copy]) # TODO: Make sure this is working properly
        # shaps = explainer(x)
        # preds = torch.sigmoid(mdl(x))
        # # shaps = ((shaps - torch.min(shaps, axis=1).values.reshape(-1,1))/(torch.max(shaps, axis=1).values.reshape(-1,1) - torch.min(shaps, axis=1).values.reshape(-1,1))) # You can't min-max per row like I do here. This causes lots of problems mathematically. But doing min-max overall takes away from the goal of making the SHAP values relative to each other. 
        # # shaps = ((abs(shaps) - torch.min(abs(shaps), axis=1).values.reshape(-1,1))/(torch.max(abs(shaps), axis=1).values.reshape(-1,1) - torch.min(abs(shaps), axis=1).values.reshape(-1,1)))*shaps.sign()
        # shaps = (((abs(shaps) / (abs(shaps).sum(axis=1)).reshape(-1,1)))*torch.sign(shaps))

        # shaps = torch.cat([shaps[:, :prot_col_idx], shaps[:, prot_col_idx+1:]], dim=1)
        # a0 = (x[:, prot_col_idx]==0).flatten()
        # a1 = (x[:, prot_col_idx]==1).flatten() 

        # if sum((preds.flatten()>=.5) & (a0)) == 0:
        #     s0_1_mean = torch.tensor(0)
        #     s0_1_sd = torch.tensor(1)
        # else:
        #     s0_1_mean = torch.mean(shaps[a0 & (preds.flatten()>=.5), :], axis=0)
        #     s0_1_sd = torch.std(shaps[a0 & (preds.flatten()>=.5), :], axis=0)
        # if sum((preds.flatten()>=.5) & (a1)) == 0:    
        #     s1_1_mean = torch.tensor(0)
        #     s1_1_sd = torch.tensor(1)
        # else:
        #     s1_1_mean = torch.mean(shaps[a1 & (preds.flatten()>=.5), :], axis=0)
        #     s1_1_sd = torch.std(shaps[a1 & (preds.flatten()>=.5), :], axis=0)
        
        # if sum((preds.flatten()<.5) & (a0)) == 0:
        #     s0_0_mean = torch.tensor(0)
        #     s0_0_sd = torch.tensor(1)
        # else:
        #     s0_0_mean = torch.mean(shaps[a0 & (preds.flatten()<.5), :], axis=0)
        #     s0_0_sd = torch.std(shaps[a0 & (preds.flatten()<.5), :], axis=0)
        # if sum((preds.flatten()<.5) & (a1)) == 0:    
        #     s1_0_mean = torch.tensor(0)
        #     s1_0_sd = torch.tensor(1)
        # else:
        #     s1_0_mean = torch.mean(shaps[a1 & (preds.flatten()<.5), :], axis=0)
        #     s1_0_sd = torch.std(shaps[a1 & (preds.flatten()<.5), :], axis=0)
        # # return torch.sum(torch.sqrt((s0_1_mean - s1_1_mean)**2) + torch.sqrt((s0_0_mean - s1_0_mean)**2))
        # return torch.sum(abs(s0_1_mean - s1_1_mean)/((s0_1_sd+s1_1_sd)/2) + abs(s0_0_mean - s1_0_mean)/((s0_0_sd+s1_0_sd)/2))
        
        
        # This is a bad idea, Can try discriminability index (abs(mu_a - mu_b) / ((s_a + s_b)/2)
        
    