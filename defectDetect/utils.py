import os

import joblib
import pandas as pd
import yaml

from defectDetect.config import artifacts_model_pickle_path, logger


def write_yaml(data: pd.DataFrame, filepath: str):
    logger.info("Writing yaml file.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def read_yaml(file_path: str):
    logger.info("Reading yamlfile")
    with open(file_path, "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params


def save_models(models, names: str):
    logger.info("Saving models into pickle files")
    os.makedirs(artifacts_model_pickle_path, exist_ok=True)
    for model, name in zip(models, names):
        filepath = os.path.join(artifacts_model_pickle_path, name + ".pkl")
        joblib.dump(model, filepath)


def save_obj(obj, filepath: str):
    logger.info("Saving objects into pickle files.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
