import torch
from metrics import *

def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return(indicator_matrix)

def neg_par_log_likelihood(pred, ytime, yevent):
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()

    risk_set_sum = ytime_indicator.mm(torch.exp(pred))
    diff = pred - torch.log(risk_set_sum + 1e-9)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)

    if n_observed == 0:
        cost = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        cost = (-sum_diff_in_observed / n_observed).reshape((-1,))

    return cost

def temp_loss(pred, ytime, yevent):
    # # log(1) = 0
    # # log(0) = -inf
    
    # # good c_index = 1
    # # bad c_index = 0
    c_index = concordance_index(pred, yevent, ytime)
    c_index_loss = -torch.log(c_index + 1e-9)
    
    # # good brier = 0
    # # bad brier = 1
    # brier = brier_score(pred, yevent)
    # brier_loss = -torch.log((1-brier) + 1e-9)
    
    # print("c_index: ", c_index)
    # print("c_index_loss: ", c_index_loss)
    # print("brier: ", brier)
    # print("brier_loss: ", brier_loss)
    
    loss = c_index_loss # + brier_loss
    
    return loss