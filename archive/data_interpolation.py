import os
import glob
import shutil
import ujson
import gzip
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
import logging
from dotenv import find_dotenv
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import json
import logging

import wandb

pd.set_option("display.max_columns", None, "display.max_rows", 200)


def delta_bin(file, enumerate_limit):
    df = pd.read_parquet(file)
    dte_dfs = []
    for en, daysToExpiration in enumerate(df.daysToExpiration.unique()):
        if en == enumerate_limit:
            break
        move_on = False

        df_expdate = df[df.daysToExpiration == daysToExpiration].copy()
        # interpolate the missing strikes
        df_expdate["delta_C"] = df_expdate["delta_C"].interpolate(
            method="linear", limit_direction="both", limit_area=None
        )
        df_expdate["delta_P"] = df_expdate["delta_P"].interpolate(
            method="linear", limit_direction="both", limit_area=None
        )

        df_expdate["delta_C"] = df_expdate["delta_C"].ffill()
        df_expdate["delta_P"] = df_expdate["delta_P"].ffill()

        df_expdate["delta_C"] = df_expdate["delta_C"].bfill()
        df_expdate["delta_P"] = df_expdate["delta_P"].bfill()

        df_expdate["delta_C"] = df_expdate["delta_C"].clip(0.001, 0.999)
        df_expdate["delta_P"] = df_expdate["delta_P"].clip(-0.001, -0.999)

        if df_expdate["delta_C"].isna().sum() == len(df_expdate):
            move_on = True

        if df_expdate["delta_P"].isna().sum() == len(df_expdate):
            move_on = True

        if move_on is False:
            put_bins = [-1.0, -0.75, -0.5, -0.25, 0.0]
            call_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            binned_df_results = []
            for pc, bins in zip(["C", "P"], [call_bins, put_bins]):

                df_expdate[f"delta_{pc}_bin"] = pd.cut(
                    df_expdate[f"delta_{pc}"], bins=bins
                )
                # binned_bidSize = df_expdate.groupby([f"delta_{pc}_bin"], observed=False)[f"bidSize_{pc}"].sum().reset_index()
                # binned_askSize = df_expdate.groupby([f"delta_{pc}_bin"], observed=False)[f"askSize_{pc}"].sum().reset_index()
                # binned_openInterest = df_expdate.groupby([f"delta_{pc}_bin"], observed=False)[f"openInterest_{pc}"].sum().reset_index().T

                for cs_pfx, cs in zip(
                    ["bs", "as", "oi"],
                    [f"bidSize_{pc}", f"askSize_{pc}", f"openInterest_{pc}"],
                ):
                    binned_df = (
                        df_expdate.groupby([f"delta_{pc}_bin"], observed=False)[cs]
                        .sum()
                        .reset_index()
                        .T
                    )
                    new_cols = binned_df.iloc[0].values
                    binned_df.columns = new_cols
                    binned_df = binned_df[1:].copy()
                    binned_df.columns = [
                        f"{en}_{cs_pfx}{pc}{col}".replace(" ", "")
                        for col in binned_df.columns
                    ]
                    binned_df.reset_index(drop=True, inplace=True)

                    binned_df_results.append(binned_df)

            binned_df_results = pd.concat(binned_df_results, axis=1)
            binned_df_results[f"{en}_expsAway"] = daysToExpiration

            dte_dfs.append(binned_df_results)

    dte_dfs = pd.concat(dte_dfs, axis=1)
    df_stock_market = (
        df[
            [
                "change",
                "percentChange",
                "close",
                "bid",
                "ask",
                "last",
                "mark",
                "markChange",
                "markPercentChange",
                "bidSize",
                "askSize",
                "highPrice",
                "lowPrice",
                "openPrice",
                "totalVolume",
                "fiftyTwoWeekHigh",
                "fiftyTwoWeekLow",
                "interestRate",
                "underlyingPrice",
                "numberOfContracts",
                "humanTime",
            ]
        ]
        .drop_duplicates()
        .reset_index()
    )
    if len(df_stock_market) != 1:
        raise ValueError("There should only be one row for each day")

    df_complete = pd.concat([dte_dfs, df_stock_market], axis=1)
    return df_complete


if __name__ == "__main__":

    files = glob.glob(
        os.path.join(os.getcwd(), "data", "processed", "SPY", "*.parquet")
    )
    enumerate_limit = 9

    comp_dfs = Parallel(n_jobs=-2, timeout=999999, backend="multiprocessing")(
        delayed(delta_bin)(file, enumerate_limit) for file in tqdm(files)
    )

    df = pd.concat(comp_dfs, axis=0)
    df.sort_values(by=["humanTime"], inplace=True)
    df['dow'] = pd.to_datetime(df['humanTime']).dt.dayofweek
    df['dom'] = pd.to_datetime(df['humanTime']).dt.day
    df['month'] = pd.to_datetime(df['humanTime']).dt.month
    
    # find outliers through iqr for close fiftyTwoWeekLow, fiftyTwoWeekLow, underlyingPrice
    for col in ['close', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'underlyingPrice']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        print(col, ":", df[(df[col]<lower_bound) | (df[col]>upper_bound)].shape[0])
        # replace outlier with nan and interpolate
        df.loc[(df[col]<lower_bound) | (df[col]>upper_bound), col] = np.nan
        df[col] = df[col].interpolate(method='linear', limit_direction='both', limit_area=None)
        
    
    # plot fiftyTwoWeekLow, fiftyTwoWeekLow, fiftyTwoWeekLow on the same plot
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(x='humanTime', y='fiftyTwoWeekLow', data=df, ax=ax, label='52 Week Low')
    sns.lineplot(x='humanTime', y='fiftyTwoWeekHigh', data=df, ax=ax, label='52 Week High')
    sns.lineplot(x='humanTime', y='underlyingPrice', data=df, ax=ax, label='underlyingPrice')
    
    plt.show()
    
    # Check if the directory exists
    directory = os.path.join(os.getcwd(), "data", "interpolated", "SPY")
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)

    df.interpolate().isna().sum().sum()
    # df[df['9_expsAway'].isna()]
    
    
    df.to_parquet(os.path.join(directory, "delta_bin.parquet"))
    
    wandb.init(project="rlot")
    #log artifact
    artifact = wandb.Artifact('delta_bin', type='data')
    artifact.add_file(os.path.join(directory, "delta_bin.parquet"))
    wandb.log_artifact(artifact)
    wandb.finish()
    
    
    
    