"""Main script for LightGBM training"""

from typing import Any, Dict, Tuple, Union

import lightgbm as lgb
import neptune
import numpy as np
import pandas as pd
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

from cancer_nn.metrics import (
    margin_loss,
    max_error_99,
    max_error_99_mape,
    max_error_mape,
    mean_absolute_percentage_error,
    percent_correct,
)


class LightGBMTrainer:
    def __init__(self, data: pd.DataFrame, max_hyperopt_evals: int = 1) -> None:
        """init function for LightGBM trainer

        Args:
        --------
            data (pd.DataFrame): data frame with data prepared for LightGBM
            max_hyperopt_evals (int, optional): hyperopt
            number of iterations. Defaults to 1.
        """
        self.data = data
        self.max_hyperopt_evals = max_hyperopt_evals
        self.best = None
        self.del_target_columns()
        (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        ) = self.train_val_test_split()
        self.test_pred = None

    def del_target_columns(self) -> None:
        """Delete columns connected
        with target time"""
        del self.data["timegap_20"]
        del self.data["time_20"]
        del self.data["dose_20"]

    def train_val_test_split(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
    ]:
        """train, val, test split

        Returns:
        --------
            (pd.DataFrame): x_train
            (pd.Series): y_train
            (pd.DataFrame): x_val
            (pd.Series): y_val
            (pd.DataFrame): x_test
            (pd.Series): y_test
        """
        x_train, x_val, x_test = (
            self.data.head(self.data.shape[0] - 40000),
            self.data.iloc[-40000:-20000],
            self.data.tail(20000),
        )
        y_train = x_train["y"]
        y_val = x_val["y"]
        y_test = x_test["y"]
        del x_train["y"]
        del x_val["y"]
        del x_test["y"]
        return (x_train, y_train, x_val, y_val, x_test, y_test)

    def hyperopt_train_test(self, params: Dict[str, Any]) -> float:
        """train lightgbm with given params
        and cross validation cv=5, optimize
        neg_mean_absolute_error metric

        Args:
        --------
            params (Dict[str, Any]): lightGBM parameters

        Returns:
        --------
            float: neg mean absolute error
            calculated on cross validation
        """
        reg = lgb.LGBMRegressor(**params)
        return cross_val_score(
            reg,
            pd.concat([self.x_train, self.x_val]),
            pd.concat([self.y_train, self.y_val]),
            scoring="neg_mean_absolute_error",
            cv=5,
        ).mean()

    def f(self, params: Dict[str, Any]) -> Dict[str, Union[float, bool]]:
        """function wich returned value w
        ill be optimized with hyperopt

        Args:
        --------
            params (Dict[str, Any]): lightGBM parameters

        Returns:
        --------
            Dict[str, Union[float, bool]]: Dict with loss
            (which will be minimized by hyperopt)
            and status (tarined with success or failure)
        """
        loss = self.hyperopt_train_test(params)
        return {"loss": (-1) * loss, "status": STATUS_OK}

    def run_hyperopt(self) -> None:
        """Run best hyperprameters
        search with hyperopt
        """
        lgb_reg_params = {
            "learning_rate": hp.uniform("learning_rate", 0.001, 0.99),
            "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 40, 2)),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 0.99),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "n_estimators": hp.choice("n_estimators", (100, 250, 500)),
            "boosting_type": hp.choice("boosting_type", ["dart", "gbdt"]),
            "num_leaves": scope.int(hp.quniform("num_leaves", 20, 300, 5)),
        }

        trials = Trials()
        self.best = fmin(
            self.f,
            lgb_reg_params,
            algo=tpe.suggest,
            max_evals=self.max_hyperopt_evals,
            trials=trials,
        )

        int_features = [
            "min_child_weight",
            "n_estimators",
            "num_leaves",
            "boosting_type",
        ]

        for feature in int_features:
            self.best[feature] = int(self.best[feature])

        self.best["boosting_type"] = ["dart", "gbdt"][self.best["boosting_type"]]
        self.best["n_estimators"] = [100, 250, 500][self.best["n_estimators"]]

    def train_final_model(self) -> None:
        """Train final model on train + val split
        create  prdictions for x_test
        """
        self.best_model = lgb.LGBMRegressor(**self.best, verbose=1)
        self.best_model.fit(
            pd.concat([self.x_train, self.x_val]),
            pd.concat([self.y_train, self.y_val]),
        )

        self.test_pred = self.best_model.predict(self.x_test)

    def log_metrics(self, neptune_configs: Dict[str, str]) -> None:
        """log metric to neptune

        Args:
        --------
            neptune_configs (Dict[str, str]):
            Neptune config with API key
            and project name
        """
        neptune.init(neptune_configs["project"], api_token=neptune_configs["api_token"])

        # create experiment
        neptune.create_experiment(name="lgbm", params=self.best, tags=["lgbm"])

        neptune.log_metric(
            "test_rmse", mean_squared_error(self.y_test, self.test_pred) ** 0.5
        )  # metrics, losses
        neptune.log_metric(
            "max_test_error", max_error(self.y_test, self.test_pred)
        )  # metrics, losses
        neptune.log_metric(
            "max_test_error_99", max_error_99(self.y_test, self.test_pred)
        )  # metrics, losses
        neptune.log_metric(
            "mape_test", mean_absolute_percentage_error(self.y_test, self.test_pred)
        )  # metrics, losses
        neptune.log_metric(
            "test_max_error_mape", max_error_mape(self.y_test, self.test_pred)
        )
        neptune.log_metric(
            "test_max_error_99", max_error_99(self.y_test, self.test_pred)
        )
        neptune.log_metric(
            "test_max_error_99_mape", max_error_99_mape(self.y_test, self.test_pred)
        )
        neptune.log_metric(
            "test_rmse",
            np.mean((np.array(self.y_test) - np.array(self.test_pred)) ** 2) ** 0.5,
        )

        neptune.log_metric("test_mae", mean_absolute_error(self.y_test, self.test_pred))

        x_test_1, y_test_1 = shuffle(self.x_test, self.y_test, random_state=0)
        test_pred_1 = self.best_model.predict(x_test_1)
        neptune.log_metric(
            "test_margin_loss",
            margin_loss(
                self.y_test.values, y_test_1.values, self.test_pred, test_pred_1
            ),
        )
        neptune.log_metric(
            "test_percentage_correct",
            percent_correct(
                self.y_test.values, y_test_1.values, self.test_pred, test_pred_1
            ),
        )
