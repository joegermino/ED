def statistical_parity(preds, A):
    '''
    ref:
    Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012, January). 
    Fairness through awareness. In Proceedings of the 3rd innovations in theoretical 
    computer science conference (pp. 214-226).
    '''
    sp = preds[A==1].sum()/preds[A==1].shape[0] - preds[A==0].sum()/preds[A==0].shape[0]
    return abs(sp)

def equalized_odds(preds, y, A):
    '''
    ref:
    Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. 
    Advances in neural information processing systems, 29.
    '''
    pos1 = preds[(A==1) & (y==1)].sum()/preds[(A==1) & (y==1)].shape[0]
    neg1 = preds[((A==0)) & (y==1)].sum()/preds[(A==0) & (y==1)].shape[0]
    pos0 = preds[(A==1) & (y==0)].sum()/preds[(A==1) & (y==0)].shape[0]
    neg0 = preds[(A==0) & (y==0)].sum()/preds[(A==0) & (y==0)].shape[0]
    eo = abs(pos1 - neg1) + abs(pos0 - neg0)
    return eo
