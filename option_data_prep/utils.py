import os
import shutil
import ujson
import time
import logging
import gzip
import json


def clean_dir(dir_to_clean: str, ticker: str = None, wait_time: int = 0):
    if "option_chain_data" in dir_to_clean:
        raise ValueError(">>>YOU ARE TRYING TO DELETE RAW DATA!!!")
    if ticker is not None:
        dir_to_clean = os.path.join(dir_to_clean, ticker)
    if os.path.exists(dir_to_clean):
        logging.info(f"Removing {dir_to_clean}")
        shutil.rmtree(dir_to_clean)
        time.sleep(wait_time)
    os.makedirs(dir_to_clean)


def unzip_ujson(file):
    with gzip.open(file, "rt", encoding="utf-8") as gz:
        data = ujson.load(gz)
    return data


def dtype_config():
    # set configs to global variable
    global dtype_configs
    with open(os.path.join(os.getcwd(), "configs", "dtypes.json")) as f:
        dtype_configs = json.load(f)

    return dtype_configs
