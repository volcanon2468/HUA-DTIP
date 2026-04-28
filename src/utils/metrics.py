import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.l1_loss(pred.float(), target.float()).item())

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.mse_loss(pred.float(), target.float()).item())

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(F.mse_loss(pred.float(), target.float())).item())

def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.float().view(-1)
    t = target.float().view(-1)
    p_centered = p - p.mean()
    t_centered = t - t.mean()
    cov = (p_centered * t_centered).mean()
    std_p = p.std(unbiased=False)
    std_t = t.std(unbiased=False)
    return float(cov / (std_p * std_t + 1e-08))

def f1(pred_labels: np.ndarray, true_labels: np.ndarray, average: str='macro') -> float:
    return float(f1_score(true_labels, pred_labels, average=average, zero_division=0))

def coverage(pred_std: torch.Tensor, pred_mean: torch.Tensor, target: torch.Tensor, z: float=1.96) -> float:
    lower = pred_mean - z * pred_std
    upper = pred_mean + z * pred_std
    within = ((target >= lower) & (target <= upper)).float()
    return float(within.mean().item())