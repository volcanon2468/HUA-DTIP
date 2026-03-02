import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred - pred.mean()
    t = target - target.mean()
    denom = torch.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-8
    return (p * t).sum().item() / denom.item()


def activity_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    truth = labels.cpu().numpy()
    return f1_score(truth, preds, average="macro", zero_division=0)


def activity_precision(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    truth = labels.cpu().numpy()
    return precision_score(truth, preds, average="macro", zero_division=0)


def activity_recall(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    truth = labels.cpu().numpy()
    return recall_score(truth, preds, average="macro", zero_division=0)


def coverage_probability(mu: np.ndarray, sigma: np.ndarray, true: np.ndarray, z: float = 1.96) -> float:
    """Fraction of true values inside [mu - z*sigma, mu + z*sigma]."""
    lo = mu - z * sigma
    hi = mu + z * sigma
    return float(np.mean((true >= lo) & (true <= hi)))
