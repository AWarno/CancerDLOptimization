"""
Script  for custom metrics functions which 
are used for experiments evaluation.
"""

import numpy as np
import torch


def margin_loss(
    y_true_1: np.array, y_true_2: np.array, y_pred_1: np.array, y_pred_2: np.array
) -> float:
    """Margin Ranking Loss (MRL):
    https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html

    Args:
    ---------------
        y_true_1 (np.array): true labels for sample 1
        y_true_2 (np.array): predictions for sample 1
        y_pred_1 (np.array): true labels for sample 2
        y_pred_2 (np.array): predictions for sample 2

    Returns:
    ---------------
        float: MRL value
    """
    y_pred_1 = torch.Tensor(y_pred_1.flatten())
    y_pred_2 = torch.Tensor(y_pred_2.flatten())
    y_true_1 = torch.Tensor(y_true_1.flatten())
    y_true_2 = torch.Tensor(y_true_2.flatten())
    order = (y_true_1 - y_true_2).sign()
    loss = torch.nn.MarginRankingLoss()
    return loss(y_pred_1, y_pred_2, order).item()


def percent_correct(
    y_true_1: np.array, y_true_2: np.array, y_pred_1: np.array, y_pred_2: np.array
) -> float:
    """
    Percentage of correctly ordered pairs i.e.
    where order of true labels is in line with
    order of predicted labels:

    Args:
    ---------------
        y_true_1 (np.array): true labels for sample 1
        y_true_2 (np.array): predictions for sample 1
        y_pred_1 (np.array): true labels for sample 2
        y_pred_2 (np.array): predictions for sample 2

    Returns:
    ---------------
        float: Percentage of correctly ordered pairs
        as fraction,  value
    """
    y_pred_1 = torch.Tensor(y_pred_1.flatten())
    y_pred_2 = torch.Tensor(y_pred_2.flatten())
    y_true_1 = torch.Tensor(y_true_1.flatten())
    y_true_2 = torch.Tensor(y_true_2.flatten())
    order_1 = (y_true_1 - y_true_2).sign()
    order_2 = (y_pred_1 - y_pred_2).sign()
    percent = (order_1 == order_2).sum() / y_true_1.size()[0]
    return percent.item()


def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    """MAPE metric

    Args:
    ---------------
        y_true (np.array): true labels
        y_pred (np.array): predicted labels

    Returns:
    ---------------
        float: MAPE value
    """
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true)))


def max_error_99(y_true: np.array, y_pred: np.array) -> float:
    """99 percentile MAE error

    Args:
    ---------------
        y_true (np.array): true labels
        y_pred (np.array): predicted labels

    Returns:
    ---------------
        float: 99 percentile MAE error value
    """
    return np.max(np.sort(np.abs(y_true - y_pred))[: int(y_true.shape[0] * 0.99)])


def max_error_mape(y_true: np.array, y_pred: np.array) -> float:
    """Max MAPE error

    Args:
    ---------------
        y_true (np.array): true labels
        y_pred (np.array): predicted labels

    Returns:
    ---------------
        float: Max MAPE error value
    """
    mape = np.abs((y_true - y_pred) / np.abs(y_true))
    return np.max(np.sort(mape))


def max_error_99_mape(y_true: np.array, y_pred: np.array) -> float:
    """99 percentile MAPE error

    Args:
    ---------------
        y_true (np.array): true labels
        y_pred (np.array): predicted labels

    Returns:
    ---------------
        float: 99 percentile MAPE error value
    """
    mape = np.abs((y_true - y_pred) / np.abs(y_true))
    return np.max(np.sort(mape)[: int(y_true.shape[0] * 0.99)])
