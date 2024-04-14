import os
import glob
import logging
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from src.option_extraction.utils import clean_dir, unzip_ujson, dtype_config
from dotenv import load_dotenv, find_dotenv
from src.option_extraction.data.data_pipeline_v2 import (
    data_inference,
    impute_peaks_rf,
)
import shutil
import time
import wandb
import glob

pd.set_option("display.max_columns", None, "display.max_rows", 200)
pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

load_dotenv(find_dotenv())


# --------------------------------------------------------------
# Save to wandb
# --------------------------------------------------------------
def save_to_wandb(folder):
    # clean all ".DS_Store" files
    for file_type in [".DS_Store"]:
        redundant_files = glob.glob(
            os.path.join(folder, "**", file_type), recursive=True
        )
        for del_file in redundant_files:
            os.remove(del_file)

    for redundant_folders in ["__pycache__"]:
        redundant_folders = glob.glob(
            os.path.join(folder, "**", redundant_folders), recursive=True
        )
        for del_folder in redundant_folders:
            shutil.rmtree(del_folder)

    wandb.init(project="rlot-data-pipeline")
    artifact = wandb.Artifact("desc_stats_nounderlying", type="data")
    artifact.add_dir(os.path.join(folder, "data", "train"),name="train")
    artifact.add_dir(os.path.join(folder, "model"), name="model")
    artifact.add_dir(os.path.join(folder, "src"), name="src")
    wandb.log_artifact(artifact)
    wandb.finish()


def check_column_order(df):
    df_exo_order = df[df.columns[df.columns.str.contains("_dte")]]
    cols_order = df_exo_order.columns
    for i in range(0, len(cols_order) - 1):
        if np.any(df_exo_order[cols_order[i]] >= df_exo_order[cols_order[i + 1]]):
            print(f"{cols_order[i]} is greater than {cols_order[i+1]}")


# --------------------------------------------------------------
# Data Scaling
# --------------------------------------------------------------


def data_scaler(df: pd.DataFrame, filename: str, train: bool = True):
    underlying_price = df.pop("underlyingPrice").to_numpy()
    human_time = df.pop("humanTime").to_numpy()
    df = df.drop(columns=["tickerPrice"])
    if train:
        df = fit_scaler(df)
    else:
        df = transform_scaler(df)

    data = []
    for en, i in enumerate(df.index):
        data.append(
            {
                "underlyingPrice": underlying_price[en],
                "humanTime": human_time[en],
                "data": df.iloc[en].to_numpy(),
            }
        )

    # save the data
    train_folder = os.path.join(folder, "data", "train", ticker)
    with open(os.path.join(train_folder, filename), "wb") as f:
        pickle.dump(data, f)


def fit_scaler(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    transform_data = scaler.fit_transform(df)
    df = pd.DataFrame(transform_data, columns=df.columns)
    scaler_file = os.path.join(folder, "model", ticker, "scaler.pkl")
    pickle.dump(scaler, open(scaler_file, "wb"), protocol=5)
    return df


def transform_scaler(df):
    scaler_file = os.path.join(folder, "model", ticker, "scaler.pkl")
    scaler = pickle.load(open(scaler_file, "rb"))
    transform_data = scaler.transform(df)
    df = pd.DataFrame(transform_data, columns=df.columns)
    return df


def split_data(df):
    test_df_front = df.iloc[:500]
    test_df_back = df.iloc[-500:]
    train_df = df.iloc[500:-500]
    return train_df, test_df_front, test_df_back


# --------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------


def pipeline(folder, ticker, file_paths, inference=False):
    if inference:
        wait_time = 0
    else:
        wait_time = 2
    data_infered_path = os.path.join(folder, "data", "interim")
    clean_dir(dir_to_clean=data_infered_path, ticker=ticker, wait_time=wait_time)
    if inference:
        if isinstance(file_paths, list):
            file_paths = file_paths[0]
        success = data_inference(file_paths, data_infered_path)
        logging.info(f"Success: {success}")

    else:
        success = Parallel(n_jobs=-1)(
            delayed(data_inference)(file, data_infered_path)
            for file in tqdm(file_paths)
        )
        logging.info(
            f"Success: {sum(success)} out of {len(success)}, Percentage: {sum(success)/len(success)}"
        )

    processed_files = glob.glob(os.path.join(data_infered_path, ticker, "*.parquet"))
    df = pd.read_parquet(processed_files, engine="pyarrow")

    logging.info(f"Shape of the dataframe: {df.shape}")
    if not inference:
        df.sort_values(by=["humanTime"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # TODO: include the this in the inference
        df = impute_peaks_rf(df)
        check_column_order(df)

    train_dir = os.path.join(folder, "data", "train")
    clean_dir(dir_to_clean=train_dir, ticker=ticker, wait_time=wait_time)
    model_dir = os.path.join(folder, "model")
    if not inference:
        clean_dir(dir_to_clean=model_dir, ticker=ticker, wait_time=wait_time)
        train_df, test_df_front, test_df_back = split_data(df)
        data_scaler(df=train_df, filename="train_data.pkl", train=True)
        data_scaler(df=test_df_front, filename="test_front_data.pkl", train=False)
        data_scaler(df=test_df_back, filename="test_back_data.pkl", train=False)
    else:
        data_scaler(df, filename="inference_data.pkl", train=False)


if __name__ == "__main__":
    RAW_DATA_ROOT = os.getenv("OPTION_DATA_LOCAL_ROOT")
    ticker = "SPY"
    file_paths = glob.glob(
        os.path.join(RAW_DATA_ROOT, ticker, "**", "*.json.gz"), recursive=True
    )
    folder = os.getcwd()

    # this will be the root folder where the data will be saved
    # for inference this will be the artifact directory
    # or copy the information from the artifact directory to this folder

    pipeline(folder, ticker, file_paths, inference=False)
    save_to_wandb(folder)
