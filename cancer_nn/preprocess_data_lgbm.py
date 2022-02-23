"""Script for data preprocessing"""

import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tqdm

DATA_PATH = "https://raw.githubusercontent.com/mkmkl93/ml-ca/master/data/uniform_200k/dataset1_200.csv"
OUTPUT_DATA_PATH = "../data/data_lgbm.csv"


def preprocess_row(idx):
    """Preprocess one series, add necessary columns"""
    row = data.iloc[idx]
    new_df = pd.DataFrame()
    time = [v for i, v in enumerate(row[1:-2]) if i % 2 == 0] + [0]
    series_len = new_df.shape[0] + 1
    dose = [v for i, v in enumerate(row[1:-1]) if i % 2 != 0] + [0]
    time_gap = [0] + list(pd.Series(time).diff().dropna().values)
    for i in range(len(time)):
        new_df[f"time_{i}"] = np.array([time[i]])

    for i in range(len(time)):
        new_df[f"dose_{i}"] = np.array([dose[i]])

    for i in range(len(time)):
        new_df[f"timegap_{i}"] = np.array([time_gap[i]])

    new_df["y"] = np.array([row[-1]])
    return new_df


def main():
    """Distributed preparation of csv with prerpocessed series"""
    print(f"Availiable CPU cores number is {mp.cpu_count() // 10}")
    k = data.shape[0]

    with Pool(mp.cpu_count() // 10) as proc:
        results = list(
            tqdm.tqdm(proc.imap_unordered(preprocess_row, range(k)), total=k)
        )

    new_data = pd.concat(results)
    new_data.to_csv(OUTPUT_DATA_PATH)
    print(f"preprocessed data saved: {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    main()
