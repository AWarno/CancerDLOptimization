"""'Main function for training 
cancer custom neural networks
"""

import click
import pandas as pd

from cancer_nn.experiments_helpers import read_yml_config
from cancer_nn.lgbm import LightGBMTrainer


@click.command()
@click.option("-csv_path", default="data/data_lgbm.csv", help="path to csv")
@click.option("-config_folder", default="configs", help="path to folder wih config")
@click.option(
    "-neptune_config",
    default="neptune_config.yml",
    help="""path to ymlfolder wih neptune project name and key""",
)
def main(csv_path: str, config_folder: str, neptune_config: str):
    """main function for training
    custom neural networks for cancer
    EMT6Ro simulation
    approximation

    Args:
    -------------
        csv_path (str): path to csv file with data
        config_folder (str): folder name with configs
        neptune_config (str): neptune yml file
        name with project name and key
    """

    neptune_config = read_yml_config(config_folder, neptune_config)
    dataset = pd.read_csv(csv_path, index_col=[0])
    LightGBM_exp = LightGBMTrainer(dataset)
    LightGBM_exp.run_hyperopt()
    LightGBM_exp.train_final_model()
    LightGBM_exp.log_metrics(neptune_config)


if __name__ == "__main__":
    main()
