"""'Pythorch lightning module for cancer optimization
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from black import out
from torch.utils.data import DataLoader

from cancer_nn.datasets import Dataset, DatasetMargin
from cancer_nn.used_networks import MultiHeadTaskRegressor, MultiTaskRegressor

LOSSES_DICT = {
    "L1": torch.nn.L1Loss(),
    "HuberLoss": torch.nn.HuberLoss(),
    "MSE": torch.nn.MSELoss(),
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class CancerNet(pl.LightningModule):
    """Pythorch lightning cancer net module"""

    def __init__(
        self,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        series_len: int = 20,
        lr: float = 0.001,
        margin_loss: bool = True,
        config_list: List[Dict[str, Any]] = [{"output_last_layer": True}],
        losses: List[str] = ["L1", "MSE", "HuberLoss"],
        main_loss: str = "L1",
        margin_loss_w: Union[float, int] = 3,
        mode: str = "unet",
        w_losses: List[Union[int, float]] = [1],
        name: str = "MultiHeadTaskRegressor",  # TODO remove
    ):
        super().__init__()
        """Init function

        Args:
        ---------
            train_df (Optional[pd.DataFrame], optional): train data frame to None.
            val_df (Optional[pd.DataFrame], optional): val data frame . Defaults to None.
            test_df (Optional[pd.DataFrame], optional): test data frame . Defaults to None.
            series_len (int, optional): _description_. series length.
            lr (float, optional): learning rate. Defaults to 0.001.
            margin_loss (bool, optional): if set to be true training with 
            margin ranking loss is run. Defaults to True.
            config_list (_type_, optional): list of subnetworks config. 
            Defaults to [{"output_last_layer": True}].
            losses (List[str], optional): losses names list from (L1, HuberLoss, MSE). 
            Defaults to ["L1", "MSE", "HuberLoss"].
            main_loss (str, optional): main loss name from (L1, HuberLoss, MSE).
            Defaults to "L1".
            margin_loss_w (Union[float, int], optional): weight of margin ranking loss. Defaults to 3.
            mode (str, optional): defines type of subnetwork (fcnn, unet, 
            cnn, lstm, cnn_lstm, cnn_lstm_att). Defaults to "unet".
            w_losses (List[Union[int, float]], optional): weigths for losses,
             first weight for main loss and the rest for losses 
             passed in losses. Defaults to [1].
        """

        self.series_len = series_len
        self.lr = lr
        self.margin_loss = margin_loss
        self.margin_loss_w = margin_loss_w
        self.mode = mode
        self.w_losses = w_losses

        # data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.prepare()

        # model
        features = self.train_set.__getitem__(0)[0]
        self.emb_size = features.size()[1]
        self.val_loss = []
        self.val_mae = []
        self.test_loss = []
        self.test_mae = []
        self.train_loss_1 = []
        self.train_mae = []
        self.val_predictions = []
        self.val_true = []
        self.test_true = []
        self.test_predictions = []
        self.train_loss = []
        self.train_mae = []
        if self.margin_loss:
            self.losses = [LOSSES_DICT[loss] for loss in losses] + [
                torch.nn.MarginRankingLoss(margin=-0.04)
            ]
            self.head_output_size = len(self.losses)
            self.model = MultiHeadTaskRegressor(config_list, mode=mode)

        else:
            self.losses = [LOSSES_DICT[loss] for loss in losses]
            self.head_output_size = len(self.losses) - 1
            self.model = MultiTaskRegressor(config_list, mode=mode)
        self.losses = [LOSSES_DICT[main_loss]] + self.losses

    @staticmethod
    def smape(y_true: np.array, y_pred: np.array) -> float:
        """SMAPE metric

        Args:
        ---------
            y_true (np.array): true labels
            y_pred (np.array): predicted labels

        Returns:
        ---------
            float: SMAPE metric value
        """
        return (
            100
            / len(y_true)
            * np.sum(
                2 * np.abs(y_pred - y_true) / (0.01 + np.abs(y_true) + np.abs(y_pred))
            )
        )

    @staticmethod
    def mape(y_true: np.array, y_pred: np.array) -> float:
        """MAPE metric

        Args:
        ---------
            y_true (np.array): true labels
            y_pred (np.array): predicted labels

        Returns:
        ---------
            float: MAPE metric value
        """
        return np.mean(np.abs((y_true - y_pred) / np.abs(y_true)))

    def prepare(self) -> None:
        """Prepare data with torch dataset,
        depends on whether margin_loss is used

        """
        if self.margin_loss:
            self.train_set = DatasetMargin(self.train_df.copy())
        else:
            self.train_set = Dataset(self.train_df.copy())
        self.scalers = self.train_set.scalers
        if self.margin_loss:
            self.val_set = DatasetMargin(self.val_df.copy(), scalers=self.scalers)
            self.test_set = DatasetMargin(self.test_df.copy(), scalers=self.scalers)
        else:
            self.val_set = Dataset(self.val_df.copy(), scalers=self.scalers)
            self.test_set = Dataset(self.test_df.copy(), scalers=self.scalers)
        self.target_scale = self.train_set.target_scale

    def train_dataloader(self) -> DataLoader:
        """prepare train dataloader

        Returns:
        ---------
            DataLoader: train dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=256,
            num_workers=7,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """prepare val dataloader

        Returns:
        ---------
            DataLoader: val dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=256,
            num_workers=7,
        )

    def test_dataloader(self) -> DataLoader:
        """prepare test dataloader

        Returns:
        ---------
            DataLoader: test dataloader
        """
        return DataLoader(
            self.test_set,
            batch_size=256,
            num_workers=7,
        )

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        """forward function

        Args:
        ---------
            x (Tuple[torch.Tensor]): Tuple
            from single batch

        Returns:
        ---------
            torch.Tensor: torch tensor w
            ith predicted values
        """
        y_pred = self.model(x)
        return y_pred

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.FloatTensor:
        """training step (see pythorch-lightning
        documenation for more information)

        Args:
        ---------
            batch (Tuple[torch.Tensor]): single batch
            batch_idx (int): batch index

        Returns:
        ---------
            torch.FloatTensor: loss value
        """
        output = self(batch)
        loss_weights = self.w_losses
        loss = torch.sum(
            torch.stack(
                [
                    self.losses[i](batch[1].squeeze(), output[:, i].squeeze())
                    * loss_weights[i]
                    for i in range(self.head_output_size)
                ],
            ),
        )

        if self.margin_loss:
            margin_loss = (
                self.losses[-1](
                    output[:, 0], output[:, -1], (batch[1] - batch[3]).sign()
                )
                * self.margin_loss_w
            )
            loss += margin_loss
            self.logger.experiment["train/margin_loss"].log(margin_loss)
            self.logger.experiment["train/correct_percent"].log(
                torch.sum(
                    (output[:, 0] - output[:, -1]).squeeze().sign()
                    == (batch[1] - batch[3]).squeeze().sign()
                )
                / batch[1].size()[0]
            )

        self.logger.experiment["train/loss_mae"].log(
            torch.mean(
                torch.abs(
                    batch[1] * self.target_scale - output[:, 0] * self.target_scale
                )
            )
        )
        self.train_loss.append(loss.item())
        self.train_mae.append(
            torch.mean(torch.abs(batch[1] - output[:, 0])).detach().cpu()
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.FloatTensor:
        """validation step (see pythorch-lightning
        documenation for more information)

        Args:
        ---------
            batch (Tuple[torch.Tensor]): single batch
            batch_idx (int): batch index

        Returns:
        ---------
            torch.FloatTensor: loss value
        """
        output = self.model(batch)
        loss = torch.sum(
            torch.stack(
                [
                    self.losses[i](batch[1].squeeze(), output[:, i].squeeze())
                    * self.w_losses[i]
                    for i in range(self.head_output_size)
                ],
            ),
        )
        if self.margin_loss:
            margin_loss = (
                self.losses[-1](
                    output[:, 0], output[:, -1], (batch[1] - batch[3]).sign()
                )
                * self.margin_loss_w
            )
            loss += margin_loss
            self.logger.experiment["val/margin_loss"].log(margin_loss)
            self.logger.experiment["val/correct_percent"].log(
                torch.sum(
                    (output[:, 0] - output[:, -1]).squeeze().sign()
                    == (batch[1] - batch[3]).squeeze().sign()
                )
                / batch[1].size()[0]
            )
        self.val_loss.append(loss.detach().cpu())
        self.val_mae.append(
            torch.mean(torch.abs(batch[1] - output[:, 0])).detach().cpu()
        )
        self.val_true.append(batch[1].flatten().detach().cpu())
        self.val_predictions.append(output[:, 0].flatten().detach().cpu())
        metric = {"val_loss": loss}
        self.logger.experiment["val/loss_mae"].log(
            torch.mean(
                torch.abs(
                    batch[1] * self.target_scale - output[:, 0] * self.target_scale
                )
            )
        )

        self.log_dict(metric)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        """validation epoch end  (see pythorch-lightning
        documenation for more information)
        """
        lr = (self.optimizers().param_groups[0]["lr"],)
        self.logger.experiment["lr"].log(lr)
        self.logger.experiment["val/epoch_loss"].log(np.mean(np.array(self.val_loss)))
        self.logger.experiment["val/epoch_mae"].log(np.mean(np.array(self.val_mae)))

        self.val_loss = []
        self.val_mae = []
        mean_mape = self.mape(
            torch.flatten(torch.cat(self.val_true)).detach().numpy(),
            torch.flatten(torch.cat(self.val_predictions)).detach().numpy(),
        )
        self.logger.experiment["val/epoch_mape"].log(np.mean(np.array(mean_mape)))
        self.val_predictions = []
        self.val_true = []

    def test_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.FloatTensor:
        """test step (see pythorch-lightning
        documenation for more information)

        Args:
        ---------
            batch (Tuple[torch.Tensor]): single batch
            batch_idx (int): batch index

        Returns:
        ---------
            torch.FloatTensor: loss value
        """
        output = self.model(batch)
        mar_loss = torch.nn.MarginRankingLoss(margin=0)
        loss = torch.sum(
            torch.stack(
                [
                    self.losses[i](batch[1].squeeze(), output[:, i].squeeze())
                    for i in range(self.head_output_size)
                ],
            ),
        )
        if self.margin_loss:
            margin_loss = (
                mar_loss(output[:, 0], output[:, -1], (batch[1] - batch[3]).sign())
                * self.margin_loss_w
            )
            loss += margin_loss
            self.logger.experiment["test/margin_loss"].log(margin_loss)
            self.logger.experiment["test/correct_percent"].log(
                torch.sum(
                    (output[:, 0] - output[:, -1]).squeeze().sign()
                    == (batch[1] - batch[3]).squeeze().sign()
                )
                / batch[1].size()[0]
            )
        self.test_loss.append(loss.detach().cpu())
        self.test_mae.append(
            torch.mean(torch.abs(batch[1] - output[:, 0])).detach().cpu()
        )
        self.test_true.append(batch[1].flatten().detach().cpu())
        self.test_predictions.append(output[:, 0].flatten().detach().cpu())
        metric = {"test_loss": loss}
        self.logger.experiment["test/loss_mae"].log(
            torch.mean(
                torch.abs(
                    batch[1] * self.target_scale - output[:, 0] * self.target_scale
                )
            )
        )

        self.log_dict(metric)
        return loss

    def test_epoch_end(self, test_step_outputs):
        """test epoch end  (see pythorch-lightning
        documenation for more information)
        """
        self.logger.experiment["test/epoch_loss"].log(np.mean(np.array(self.test_loss)))
        self.logger.experiment["test/epoch_mae"].log(np.mean(np.array(self.test_mae)))
        self.test_loss = []
        self.test_mae = []
        mean_mape = self.mape(
            torch.flatten(torch.cat(self.test_true)).detach().numpy(),
            torch.flatten(torch.cat(self.test_predictions)).detach().numpy(),
        )
        self.logger.experiment["test/epoch_mape"].log(np.mean(np.array(mean_mape)))

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Adam], List[Dict[str, Any]]]:
        """configure optimizers  (see pythorch-lightning
        documenation for more information)

        Returns:
        ---------
            (Tuple(List[torch.optim], List[Dict[str, Any]])): List
            with optimizers, List with schedulers
        """
        optimizers = [torch.optim.Adam(self.parameters(), lr=self.lr)]
        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[0], mode="min", factor=0.7, patience=5, verbose=True
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        ]
        return (optimizers, schedulers)
