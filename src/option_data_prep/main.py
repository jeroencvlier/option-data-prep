# Built in libraries
import os
import glob
import logging
import pickle
from collections import OrderedDict
import shutil
import glob

# 3rd party libraries
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import wandb
from dotenv import load_dotenv, find_dotenv

# Custom libraries
from src.option_data_prep.utils import clean_dir
from option_data_prep.data.data_pipeline import data_inference, impute_peaks_rf


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
    artifact.add_dir(os.path.join(folder, "data", "train"),name="data")
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
    df.sort_values(by=["humanTime"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    human_time = df.pop("humanTime").to_numpy()
    underlying_price = df.pop("underlyingPrice").to_numpy()
    df = df.drop(columns=["tickerPrice", ])
    # create a tuple of the column structure
    scaled_columns = tuple(df.columns)
    
    if train:
        df = fit_scaler(df)
    else:
        df = transform_scaler(df)
        
    # append the human time and underlying price
    df.insert(0, "humanTime", human_time)
    df.insert(0, "underlyingPrice", underlying_price)

    data = OrderedDict()
    for index, row in df.iterrows():
        data[index] = {
            "underlyingPrice": row["underlyingPrice"],
            "humanTime": row["humanTime"],
            "data": row.drop(labels=["underlyingPrice", "humanTime"]).to_numpy(),
        }

    # save the data
    train_folder = os.path.join(folder, "data", "train", ticker)
    with open(os.path.join(train_folder, filename), "wb") as f:
        pickle.dump(data, f)
    # save the column tuple structure to model folder
    model_folder = os.path.join(folder, "model", ticker)
    # with open(os.path.join(model_folder, "columns.pkl"), "wb") as f:
    #     pickle.dump(scaled_columns, f)
    pickle.dump(scaled_columns, open(model_folder, "wb"), protocol=5)
    


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
    test_df_front = df.iloc[:500].copy()
    test_df_back = df.iloc[-500:].copy()
    train_df = df.iloc[500:-500].copy()
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
    processed_files = sorted(processed_files)
    df = pd.read_parquet(processed_files, engine="pyarrow")
    # drop duplicated timestamps
    # slice timestamps above 16h
    df['hour'] = df['humanTime'].apply(lambda x: int(x.split(" ")[1].split(":")[0])) 
    logging.info("Dropping %s duplicated timestamps, %s percentage of timestamps above 16h", len(df[df['hour']>16]), round(len(df[df['hour']>16])/len(df)*100,3))
    df = df[df['hour']<=16]
    df = df.drop(columns=['hour'])
    
    time_stamps = df["humanTime"]
    logging.info(f"Number of unique timestamps: {len(time_stamps.unique())}")
    logging.info(f"Number of duplicated timestamps: {len(time_stamps) - len(time_stamps.unique())}")    
    logging.info(f"Shape of the dataframe: {df.shape}")
    
    train_dir = os.path.join(folder, "data", "train")
    clean_dir(dir_to_clean=train_dir, ticker=ticker, wait_time=wait_time)
    model_dir = os.path.join(folder, "model")    
    if not inference:
        clean_dir(dir_to_clean=model_dir, ticker=ticker, wait_time=wait_time)
        # TODO: include the this in the inference
        df = impute_peaks_rf(df)
        check_column_order(df)        
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


    pipeline(folder, ticker, file_paths, inference=False)
    save_to_wandb(folder)


