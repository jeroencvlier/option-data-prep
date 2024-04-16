import os
import numpy as np
import pandas as pd
import logging
import warnings
import wandb
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.signal import find_peaks, peak_prominences, peak_widths
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import re

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)


from src.utils import unzip_ujson, dtype_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

configs = dtype_config()
# --------------------------------------------------------------
# Peak Discovery
# --------------------------------------------------------------


def peak_discovery(values, exp_away, col_prefix, pc, total_peaks=3):

    # Assuming 'values' is an array of your data
    hist, bin_edges = np.histogram(values, bins=200, density=True, range=(-1, 1))
    # Find peaks (local maxima) in the histogram data
    peaks, properties = find_peaks(hist, height=0)
    # Calculate the prominences of each peak
    prominences = peak_prominences(hist, peaks)[0]
    # Calculate the widths of each peak
    widths = peak_widths(hist, peaks, rel_height=0.5)[0]
    # Calculate bin width for volume calculation
    bin_width = np.diff(bin_edges)[0]

    # Calculate total volume under each peak
    peak_volumes = []
    for peak, width in zip(peaks, widths):  # Ensure iteration over widths[0]
        start_bin = int(np.max([peak - width / 2, 0]))
        end_bin = int(np.min([peak + width / 2, len(hist) - 1]))
        peak_volume = np.sum(hist[start_bin : end_bin + 1]) * bin_width
        peak_volumes.append(peak_volume)

    # Sort peaks by their height to find the highest peaks
    sorted_indices_by_height = np.argsort(properties["peak_heights"])[
        ::-1
    ]  # Indices sorted by height

    top_indices = sorted_indices_by_height[:total_peaks]  # Take top 5

    # Extract bin edges for the top 5 peaks
    highest_peaks_bins = [bin_edges[peaks[i]] for i in top_indices]
    highest_peaks_volumes = [peak_volumes[i] for i in top_indices]

    # if not 5 in list replace with nan
    if len(sorted_indices_by_height) < total_peaks:
        for i in range(total_peaks - len(sorted_indices_by_height)):
            highest_peaks_bins.append(np.nan)
            highest_peaks_volumes.append(np.nan)
    # Reporting
    peak_report = {}
    for en, peak_idx in enumerate(highest_peaks_volumes, start=0):
        peak_report.update(
            {
                f"e{exp_away}_{col_prefix}_{pc}_Pk{en+1}": np.round(
                    highest_peaks_bins[en], 6
                ),
                f"e{exp_away}_{col_prefix}_{pc}_PkV{en+1}": np.round(
                    highest_peaks_volumes[en], 6
                ),
            }
        )

    return peak_report


# --------------------------------------------------------------
# Binning Contracts
# --------------------------------------------------------------


def binning_contracts_stats(
    df: pd.DataFrame, bin_column: str, underlyingPrice: float, pc: str, exp_away: int
):
    # df = put_df
    # bin_column = "openInterest"
    # pc = "P"
    # exp_away = 0

    if bin_column in ["bidSize", "askSize", "totalVolume", "openInterest"]:
        pfxs = {
            "bidSize": "bS",
            "askSize": "aS",
            "totalVolume": "tV",
            "openInterest": "oI",
        }
        col_prefix = pfxs[bin_column]

    else:
        col_prefix = bin_column

    df_bin = df[["strikePrice", bin_column]].copy()
    # convert strike price to a percentage from underlying
    df_bin["strikePriceNorm"] = df_bin["strikePrice"] - underlyingPrice
    df_bin["strikePricePct"] = df_bin["strikePriceNorm"] / underlyingPrice

    strike_prices_norm = df_bin["strikePricePct"].values
    open_interests = df_bin[bin_column].values.astype(int)
    values = np.repeat(strike_prices_norm, open_interests)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    # mode alternative
    # frequency_count = Counter(values)
    # most_common = frequency_count.most_common(5)

    # Add descriptive statistics to the dataframe
    stats = {
        # f"e{exp_away}_{col_prefix}_{pc}_Sum": len(values),
        # f"e{exp_away}_{col_prefix}_{pc}_Mean": np.round(np.mean(values), 6),
        # f"e{exp_away}_{col_prefix}_{pc}_Med": np.round(np.median(values), 6),
        # f"e{exp_away}_{col_prefix}_{pc}_Mode": mode(values)[0],
        # f"e{exp_away}_{col_prefix}_{pc}_Mode1": most_common[0][0],
        # f"e{exp_away}_{col_prefix}_{pc}_Mode2": most_common[1][0],
        # f"e{exp_away}_{col_prefix}_{pc}_Mode3": most_common[2][0],
        # f"e{exp_away}_{col_prefix}_{pc}_Mode4": most_common[3][0],
        # f"e{exp_away}_{col_prefix}_{pc}_Mode5": most_common[4][0],
        f"e{exp_away}_{col_prefix}_{pc}_Std": np.round(np.std(values, ddof=1), 6),
        f"e{exp_away}_{col_prefix}_{pc}_Var": np.round(np.var(values, ddof=1), 6),
        f"e{exp_away}_{col_prefix}_{pc}_Range": np.round(np.ptp(values), 6),
        f"e{exp_away}_{col_prefix}_{pc}_Q1": np.round(q1, 6),
        f"e{exp_away}_{col_prefix}_{pc}_Q3": np.round(q3, 6),
        f"e{exp_away}_{col_prefix}_{pc}_IQR": np.round(q3 - q1, 6),
        # f"e{exp_away}_{col_prefix}_Skew": np.round(skew(values), 6),
        f"e{exp_away}_{col_prefix}_{pc}_Kurt": np.round(
            kurtosis(values, fisher=True), 6
        ),
        # f"e{exp_away}_{col_prefix}_Min": np.min(values),
        # f"e{exp_away}_{col_prefix}_Max": np.max(values),
    }

    peak_report = peak_discovery(values, exp_away, col_prefix, pc, total_peaks=3)

    combined_stats = {**peak_report, **stats}

    return pd.DataFrame(combined_stats, index=[0])


def data_inference(file, folder, max_expirations=10):
    try:
        data = unzip_ujson(file)
        if data["status"] == "SUCCESS":
            underlyingPrice = np.float32(data["underlyingPrice"])

            putExpDateMap = data.pop("putExpDateMap")
            callExpDateMap = data.pop("callExpDateMap")

            # align the data from the top level to the underlying level
            underlying = data.pop("underlying")
            aligned_data = {**underlying, **data}
            alighed_df = pd.DataFrame(aligned_data, index=[0])
            alighed_df = alighed_df[list(dtype_config()["underlying_data_keep"].keys())]
            alighed_df = alighed_df.rename(columns={"symbol": "ticker"})
            alighed_df["tickerPrice"] = underlyingPrice
            # Copnvert quoteTime to human readable time without miliseconds
            alighed_df["quoteTime"] = pd.to_datetime(
                alighed_df["quoteTime"], unit="ms", utc=True
            )
            alighed_df["quoteTime"] = alighed_df["quoteTime"].dt.tz_convert(
                "America/New_York"
            )
            alighed_df["humanTime"] = alighed_df["quoteTime"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            # alighed_df["hour"] = alighed_df["quoteTime"].dt.hour
            alighed_df["dow"] = alighed_df["quoteTime"].dt.dayofweek
            alighed_df["sin_dow"] = np.sin(2 * np.pi * alighed_df["dow"] / 7)
            alighed_df["cos_dow"] = np.cos(2 * np.pi * alighed_df["dow"] / 7)

            # sin and cos day of the year for season
            alighed_df["sin_doy"] = np.sin(
                2 * np.pi * alighed_df["quoteTime"].dt.dayofyear / 365
            )
            alighed_df["cos_doy"] = np.cos(
                2 * np.pi * alighed_df["quoteTime"].dt.dayofyear / 365
            )
            alighed_df.drop(
                columns=["quoteTime", "interval", "isDelayed", "ticker"], inplace=True
            )

            # alighed_df['spread'] = round(alighed_df['ask'] - alighed_df['bid'],4)

            alighed_df.drop(
                columns=[
                    "change",
                    "last",
                    "close",
                    "bid",
                    "ask",
                    "mark",
                    "markChange",
                    "markPercentChange",
                    "highPrice",
                    "lowPrice",
                    "openPrice",
                    "fiftyTwoWeekHigh",
                    "fiftyTwoWeekLow",
                    "dow",
                ],
                inplace=True,
            )
            alighed_df.drop(
                columns=[
                    "percentChange",
                    "bidSize",
                    "askSize",
                    "totalVolume",
                    "numberOfContracts",
                    "interestRate",
                ],
                inplace=True,
            )

            # min expirations available are 26
            # some might have no into so lets start with 15 expirations
            # we will use the same expirations for both calls and puts
            # first we will double check that the expirations are the same for both calls and puts
            call_expirations = set(callExpDateMap.keys())
            assert call_expirations == set(
                putExpDateMap.keys()
            ), "expirations are not the same"
            valid_expirations = []
            for expiration in sorted(call_expirations):
                # print(expiration)
                # break
                assert set(callExpDateMap[expiration].keys()) == set(
                    putExpDateMap[expiration].keys()
                ), "strikes are not the same"
                call_exp_dict = [d[0] for d in callExpDateMap[expiration].values()]
                put_exp_dict = [d[0] for d in putExpDateMap[expiration].values()]

                call_df = pd.DataFrame(call_exp_dict)
                put_df = pd.DataFrame(put_exp_dict)

                if (
                    call_df["openInterest"].sum() > 100
                    and put_df["openInterest"].sum() > 100
                ):
                    call_df = call_df[configs["strike_level_columns"].keys()].astype(
                        configs["strike_level_columns"]
                    )
                    put_df = put_df[configs["strike_level_columns"].keys()].astype(
                        configs["strike_level_columns"]
                    )
                    transposed_dfs = []
                    for bin_col in [
                        "bidSize",
                        "askSize",
                        "openInterest",
                        # "totalVolume",
                    ]:
                        transposed_dfs.append(
                            binning_contracts_stats(
                                call_df,
                                bin_col,
                                underlyingPrice,
                                "C",
                                len(valid_expirations),
                            )
                        )
                        transposed_dfs.append(
                            binning_contracts_stats(
                                put_df,
                                bin_col,
                                underlyingPrice,
                                "P",
                                len(valid_expirations),
                            )
                        )

                        # print(len(valid_expirations),int(expiration.split(":")[1]),expiration)
                    transposed_dfs = pd.concat(transposed_dfs, axis=1)
                    transposed_dfs[f"e{len(valid_expirations)}_dte"] = int(
                        expiration.split(":")[1]
                    )
                    valid_expirations.append(transposed_dfs)

                # break when a valid 15 expirations are found
                if len(valid_expirations) == max_expirations:

                    valid_expirations = pd.concat(valid_expirations, axis=1)
                    final_df = pd.concat([alighed_df, valid_expirations], axis=1)
                    filename = file.split("/")[-1].replace(".json.gz", ".parquet")

                    final_df.to_parquet(
                        os.path.join("data", folder, data["symbol"], filename)
                    )
                    success = True
                    break
        else:
            success = False
    except Exception as e:
        logging.error(f"Error in {file}: {e}")
        success = False

    return success


# --------------------------------------------------------------
# Stationary data
# --------------------------------------------------------------
def make_stationary(df_stat):

    from statsmodels.tsa.stattools import adfuller

    # Assuming 'df' is your DataFrame with a 'price' column for stock prices

    # Log Transformation
    df_stat["log_price"] = np.log(df_stat["underlyingPrice"])
    # plot the log price
    plt.plot(df_stat["log_price"])
    plt.show()

    # First Differencing
    # insert the value as 0
    df_stat["log_price_diff"] = df_stat["log_price"].diff()
    plt.plot(df_stat["log_price_diff"])
    plt.show()
    # Drop NA (created by differencing)
    df_clean = df_stat.dropna()
    # Perform ADF test
    result = adfuller(df_clean["log_price_diff"])
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])

    # Interpretation
    if result[1] < 0.05:
        logging.info("Data is stationary")
    else:
        logging.info("Data is not stationary and may need further processing")

    df_clean = df_clean.drop(columns=["log_price", "underlyingPrice"])
    return df_clean


# --------------------------------------------------------------
# Shift the data
# --------------------------------------------------------------
def shift_data(df_toshift, periods_to_shift=1):
    if periods_to_shift > 1:
        env_keep = df_toshift[["tickerPrice", "humanTime"]].copy()
        df_toshift = df_toshift.drop(columns=["tickerPrice", "humanTime"]).copy()
        shifted_list = [env_keep, df_toshift.add_prefix(f"tn0_")]
        for t_shift in range(1, periods_to_shift):
            shifted_list.append(df_toshift.shift(t_shift).add_prefix(f"tn{t_shift}_"))
        shifted_df = pd.concat(shifted_list, axis=1).dropna()
    else:
        shifted_df = df_toshift.copy()
    return shifted_df


# --------------------------------------------------------------
# Save to wandb
# --------------------------------------------------------------
def save_to_wandb(df, file_path):
    try:
        df.drop(columns=["underlyingPrice"], inplace=True)
    except:
        pass
    finally:
        df.to_parquet(file_path)
        wandb.init(project="rlot")
        artifact = wandb.Artifact("desc_stats_nounderlying", type="data")
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
        wandb.finish()


# --------------------------------------------------------------
# Peak Imputation
# --------------------------------------------------------------


def impute_peaks_rf(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("NA values: %s", df.isna().sum().sum())

    # regular expression to find all the features starts wtih "e" then a wild card then "_"
    exp_fts = set([x.split("_")[0] for x in df.columns if re.match(r"e\d_.*", x)])
    for rxp_ft in tqdm(exp_fts):
        exp_df = df.filter(regex=rf"{rxp_ft}_.*")
        layers = set(
            [x.rsplit("_", 1)[0] for x in exp_df.columns if len(x.split("_")) == 4]
        )
        for layer in tqdm(layers):
            layer_df = exp_df.filter(regex=rf"{layer}_.*")
            peak_cols = set(
                [x for x in layer_df.columns if re.match(r"Pk.*\d", x.split("_")[-1])]
            )
            # remove all peak columns and peak volume columns
            layer_df_pk = layer_df.drop(columns=peak_cols)
            assert layer_df_pk.isna().sum().sum() == 0
            for peak_col in peak_cols:
                # split test train data by seperating peak nan values and keep indexing
                test_index = layer_df[layer_df[peak_col].isna()].index
                if len(test_index) > 0:
                    test_df = layer_df_pk.loc[test_index]
                    train_df = layer_df_pk.drop(index=test_index)
                    train_target_df = layer_df[peak_col].drop(index=test_index)
                    # fit model
                    rf = RandomForestRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1
                    )
                    rf.fit(train_df, train_target_df)
                    # predict and insert into origional df on index
                    pred = rf.predict(test_df)
                    df.loc[test_index, peak_col] = pred

    logging.info("NA values: %s", df.isna().sum().sum())
    return df
