"""Script for torch dataset classes"""

import random
from pickletools import int4
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class Dataset(torch.utils.data.Dataset):
    """Cancer torch dataset maker"""

    def __init__(
        self,
        data: pd.DataFrame,
        columns: List[str] = ["time", "dose", "time_gap"],
        target_column: str = "target",
        scalers: Optional[
            Dict[str, StandardScaler]
        ] = None,  # for training, for validation the same scalers as for training should be used
    ) -> None:
        """Init function

        Args:
        ------------
            data (pd.DataFrame): data frame witch protocls and target column
            columns (List[str], optional): list with features columns
            which will be taken to model. Defaults to ["time", "dose", "time_gap"].
            target_column (str, optional): target column name. Defaults to "target".
            scalers (Optional[ List[StandardScaler] ], optional): Dict with
            feature_name: fitted StandardScaler or None (new StandardScalers
            will be fitted). Defaults to None.
        """
        self.data = data
        self.columns = columns
        self.target_column = target_column
        self.indices = list(set(data.series.values))
        self.dataset_len = len(self.indices)
        self.series_len = 21
        self.target_scale = 500
        if not scalers:
            self.scalers = {
                column: StandardScaler().fit(self.data[column].values.reshape(-1, 1))
                for column in self.columns
            }
        else:
            self.scalers = scalers
        for column in self.columns:
            self.data[column] = self.scalers[column].transform(
                self.data[column].values.reshape(-1, 1)
            )
        self.data[self.target_column] = (
            self.data[self.target_column] / self.target_scale
        )

    def __len__(self) -> int:
        """Denotes the total number of samples

        Returns:
        ------------
            int: number of samples
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Select ith series

        Args:
        ------------
            idx (int): sample index

        Returns:
        ------------
            Tuple[torch.FloatTensor, torch.FloatTensor]: (input_features, target)
        """
        x = self.data.iloc[
            self.series_len * idx : (self.series_len * idx) + self.series_len - 1
        ][self.columns].to_numpy()
        y = self.data.iloc[
            (self.series_len * idx)
            + self.series_len
            - 1 : (self.series_len * idx)
            + self.series_len
        ][self.target_column].to_numpy()
        return (torch.FloatTensor(x.astype(float)), torch.FloatTensor(y.astype(float)))


class DatasetMargin(Dataset):
    "Cancer torch dataset maker"

    def get_from_idx(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Select ith series

        Args:
        ------------
            idx (int): sample index

        Returns:
        ------------
            Tuple[torch.FloatTensor, torch.FloatTensor]: (input_features, target)
        """
        x = self.data.iloc[
            self.series_len * idx : (self.series_len * idx) + self.series_len - 1
        ][self.columns].to_numpy()
        y = self.data.iloc[
            (self.series_len * idx)
            + self.series_len
            - 1 : (self.series_len * idx)
            + self.series_len
        ][self.target_column].to_numpy()
        return (
            torch.FloatTensor(x.astype(float)),
            torch.FloatTensor(y.astype(float)),
        )

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Select series by given idx
        and rand another sample

        Args:
        ------------
            idx (int): series index

        Returns:
        ------------
            Tuple[ torch.FloatTensor, torch.FloatTensor,
            torch.FloatTensor, torch.FloatTensor ]: (input_features, target,
            other_sample_input_features, other_sample_target)
        """
        x = self.data.iloc[
            self.series_len * idx : (self.series_len * idx) + self.series_len - 1
        ][self.columns].to_numpy()
        y = self.data.iloc[
            (self.series_len * idx)
            + self.series_len
            - 1 : (self.series_len * idx)
            + self.series_len
        ][self.target_column].to_numpy()
        another_sample = self.get_from_idx(random.randint(0, self.dataset_len - 1))
        return (
            torch.FloatTensor(x.astype(float)),
            torch.FloatTensor(y.astype(float)),
            another_sample[0],
            another_sample[1],
        )
