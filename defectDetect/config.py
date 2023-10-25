import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

artifacts_model_param_path = os.path.join((Path(__file__).parent.parent), "artifacts", "params.yaml")
artifacts_results_path = os.path.join((Path(__file__).parent.parent), "artifacts", "results.yaml")
artifacts_tuned_model_param_path = os.path.join((Path(__file__).parent.parent), "artifacts", "tune", "tuned_params.yaml")
artifacts_tuned_results_path = os.path.join((Path(__file__).parent.parent), "artifacts", "tune", "tuned_results.yaml")
artifacts_model_pickle_path = os.path.join((Path(__file__).parent.parent), "artifacts", "models")
artifacts_transformer_path = os.path.join((Path(__file__).parent.parent), "artifacts", "transformer.pkl")


@dataclass
class DataConfig:
    raw_data_path = os.path.join((Path(__file__).parent.parent), "dataset", "raw", "jm1.csv")
    train_data_path = os.path.join((Path(__file__).parent.parent), "artifacts", "train.csv")
    test_data_path = os.path.join((Path(__file__).parent.parent), "artifacts", "test.csv")


@dataclass
class ModelConfig:
    weight_range = np.arange(0.1, 0.5, 0.01)
    voting = "soft"
    model_param_path = os.path.join((Path(__file__).parent.parent), "artifacts", "params.yaml")
