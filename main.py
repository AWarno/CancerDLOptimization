"""'Main function for training 
cancer custom neural networks
"""

import click
import pytorch_lightning as pl
import torch
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from cancer_nn.experiments_helpers import (
    log_custom_metrics,
    log_experiment_parameters,
    read_subnetworks_configs,
    read_yml_config,
    set_seeds,
    train_val_test_split,
)
from cancer_nn.torch_lightning_modules import CancerNet


@click.command()
@click.option("-csv_path", default="data/data.csv", help="path to csv")
@click.option(
    "-group_id",
    default="series",
    help="""column name with series id""",
)
@click.option(
    "-saved_model_path",
    default="/home/awarno/cancer/pretrained_models1",
    help="""path to folder where trained model will be saved""",
)
@click.option("-config_folder", default="configs", help="path to folder wih config")
@click.option(
    "-neptune_config",
    default="neptune_config.yml",
    help="""path to ymlfolder wih neptune project name and key""",
)
@click.option("-max_epochs", default=1, help="max epochs")
@click.option(
    "-config",
    default="config.yml",
    help="model config file",
)
@click.option("-seed", default=0, help="seed")
def main(
    csv_path: str,
    group_id: str,
    saved_model_path: str,
    config_folder: str,
    neptune_config: str,
    max_epochs: int,
    config: str,
    seed: int,
):
    """main function for training
    custom neural networks for cancer
    EMT6Ro simulation
    approximation

    Args:
    -------------
        csv_path (str): path to csv file with data
        group_id (str): column name from data from csv_path
        with series id (i.e. single protocol)
        saved_model_path (str): path to folder where
        pretrained models will be saved
        config_folder (str): folder name with configs
        neptune_config (str): neptune yml file
        name with project name and key
        max_epochs (int): max epochs
        config (str): etwork main config
        yml file name
        seed (int): seed
    """

    set_seeds(seed)
    train, val, test = train_val_test_split(csv_path, group_id)

    config = read_yml_config(config_folder, config)
    neptune_config = read_yml_config(config_folder, neptune_config)
    neptune_logger = NeptuneLogger(
        **neptune_config,
        tags=[config["network"]["config_list"][0]],
    )

    log_experiment_parameters(neptune_logger, config, seed)

    config["network"]["config_list"] = read_subnetworks_configs(config_folder, config)

    network = CancerNet(train_df=train, val_df=val, test_df=test, **config["network"])
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=12, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=saved_model_path,
        filename=f"{neptune_logger.run._short_id}.ckpt",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        default_root_dir=saved_model_path,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        auto_lr_find=False,
        logger=neptune_logger,
        gradient_clip_val=0.3,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(network)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    print(f"Downloading best weights from {best_model_path} ...")
    network = CancerNet.load_from_checkpoint(
        checkpoint_path=best_model_path,
        train_df=train,
        val_df=val,
        test_df=test,
        **config["network"],
    )
    trainer.test(network)
    log_custom_metrics(network, neptune_logger)
    print(f"Done!")


if __name__ == "__main__":
    main()
