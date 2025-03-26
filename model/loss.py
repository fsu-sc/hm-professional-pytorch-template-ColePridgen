import torch.nn.functional as F
import torch.nn as nn

mse = nn.MSELoss()
mae = nn.L1Loss()

def nll_loss(output, target):
    return F.nll_loss(output, target)
