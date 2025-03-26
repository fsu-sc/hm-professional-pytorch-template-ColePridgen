import torch
import torch.nn.functional as F

def mse(y_pred, y_true):
    return F.mse_loss(y_pred, y_true).item()

def mae(y_pred, y_true):
    return F.l1_loss(y_pred, y_true).item()

def rmse(y_pred, y_true):
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()

def r2_score(y_pred, y_true):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total).item()

# Dictionary of metric functions
metric_functions = {
    'mse': mse,
    'mae': mae,
    'rmse': rmse,
    'r2': r2_score
}
