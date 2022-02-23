"""
Script for helpers for experiments setup:
* settings seeds
* train/val/test split etc.
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle

from cancer_nn.metrics import (
    margin_loss,
    max_error_99,
    max_error_99_mape,
    max_error_mape,
    mean_absolute_percentage_error,
    percent_correct,
)
from cancer_nn.torch_lightning_modules import CancerNet


def set_seeds(seed: int) -> None:
    """set seed to torch,
    numpy, random and cuda
    Warning: perfect reproducibility
    on cuda is not achieved

    Args:
    ----------
        seed (int): seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_val_test_split(
    csv_path: str, group_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """read csv data and splits it into train, val and test
    wit ratio 8: 1: 1

    Args:
    ----------
        csv_path (str): path to csv with data
        group_id (str): series id column name

    Returns:
    ----------
        Tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame):
        train, val, test data frames
    """
    dataset = pd.read_csv(csv_path)
    n = dataset.series.max()
    train = dataset[dataset[group_id].apply(lambda x: int(x) < int(n * 0.8))]
    val = dataset[
        dataset[group_id].apply(
            lambda x: (int(x) > int(n * 0.8)) and (int(x) < int(n * 0.9))
        )
    ]
    test = dataset[dataset[group_id].apply(lambda x: int(x) > int(n * 0.9))]
    return (train, val, test)


def log_experiment_parameters(
    neptune_logger: NeptuneLogger, config: Dict[str, Any], seed: int
):
    """Log parameters to neptune

    Args:
    ------------
        neptune_logger (NeptuneLogger): neptune logger
        config (Dict[str, Any]): experiment config
        seed (int): used seed
    """
    parameters = {}

    for key in config["network"]:
        parameters[key] = config["network"][key]

    parameters["seed"] = seed

    neptune_logger.experiment["parameters"] = parameters


def read_yml_config(dir: str, file_name: str) -> Optional[Dict[str, Any]]:
    """yaml safe load

    Args:
    ------------
        dir (str): file directory
        file_name (str): file name
    Returns:
    ------------
        Optional(Dict(str, Any)): loaded yml
        file if finished with success
    """
    with open(os.path.join(dir, file_name), "r") as stream:
        try:
            yml_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
    return yml_config


def read_subnetworks_configs(
    config_folder: str, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """read subnetworks configs

    Args:
    ------------
        config_folder (str): folder with configs name
        config (Dict[str, Any]): config yml file name

    Returns:
    ------------

        List[Dict[str, Any]]: List of
        dowlnoaded config subnetworks
    """
    config_list = []
    for config_name in config["network"]["config_list"]:
        net_config = read_yml_config(config_folder, config_name)
        config_list.append(net_config)
    return config_list


def log_custom_metrics(
    network: CancerNet,
    neptune_logger: NeptuneLogger,
) -> None:
    """log custom metrics after training

    Args:
    ------------
        network (CancerNet): pl.Lighning cancer module
        neptune_logger (NeptuneLogger): neptune logger
    """
    print(f"Calculating margin loss and correct percent ...")
    y_true, y_pred = (
        np.array(torch.cat(network.test_true)) * 500,
        np.array(torch.cat(network.test_predictions)) * 500,
    )
    y_true_1, y_pred_1 = shuffle(y_true, y_pred, random_state=0)
    neptune_logger.log_metrics(
        {"test_margin": margin_loss(y_true, y_true_1, y_pred, y_pred_1)}
    )
    neptune_logger.log_metrics(
        {"test_percent": percent_correct(y_true, y_true_1, y_pred, y_pred_1)}
    )

    neptune_logger.log_metric(
        "test_rmse", mean_squared_error(y_true, y_pred) ** 0.5
    )  # metrics, losses
    neptune_logger.log_metric(
        "max_test_error", max_error(y_true, y_pred)
    )  # metrics, losses
    neptune_logger.log_metric(
        "max_test_error_99", max_error_99(y_true, y_pred)
    )  # metrics, losses
    neptune_logger.log_metric(
        "mape_test", mean_absolute_percentage_error(y_true, y_pred)
    )  # metrics, losses
    neptune_logger.log_metric("test_max_error_mape", max_error_mape(y_true, y_pred))
    neptune_logger.log_metric("test_max_error_99", max_error_99(y_true, y_pred))
    neptune_logger.log_metric(
        "test_max_error_99_mape", max_error_99_mape(y_true, y_pred)
    )
    neptune_logger.log_metric("test_rmse", np.mean((y_true - y_pred) ** 2) ** 0.5)
    neptune_logger.log_metric("test_mae", mean_absolute_error(y_true, y_pred))
