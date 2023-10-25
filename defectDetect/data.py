import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from defectDetect import config, utils


def load_dataset(filepath: str, print_info: bool = False) -> pd.DataFrame:
    """load data from source into a Pandas DataFrame.

    Args:
        filepath (str): file location.
        print_info (bool): Whether to print info and description.

    Returns:
        pd.DataFrame: Pandas DataFrame of the data.
    """
    df = pd.read_csv(filepath)
    if print_info:
        print(df.info, "\n")
        print("=" * 150)
        print("\n", df.describe().T)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data containing non numeric characters

    Args:
        df (pd.DataFrame): Data to be cleaned.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    # replace '?' with np.nan as they show the same meaning of not knowing the data value.
    df = df.map(lambda x: np.nan if x == "?" else x)
    df.fillna(0, inplace=True)

    for col in df.columns[:-1]:
        df[col] = df[col].astype(float)

    try:
        df["defects"] = df["defects"].astype(int)
    except:
        pass

    return df


def feature_engineering(df: pd.DataFrame, drop_features: bool = True) -> pd.DataFrame:
    """Feature engineering by creating features and dropping unwanted features (feature selection).

    Args:
        df (pd.DataFrame): Input data.
        drop_features (bool, optional): Whether to drop unwanted features. !!(hard coded to drop 2 features: "n", "locCodeAndComment")

    Returns:
        pd.DataFrame: New data with added features.
    """
    # feature creations
    df["complex_by_line"] = (df["v(g)"] + df["ev(g)"] + df["iv(g)"]) / df["loc"]
    df["code_by_cmtblank"] = df["lOCode"] / (df["lOComment"] + df["lOBlank"])
    df["lines"] = (df["loc"] + df["lOCode"]) / 2
    df["Opratio"] = df["uniq_Op"] / df["total_Op"]
    df["Opndratio"] = df["uniq_Opnd"] / df["total_Opnd"]

    if drop_features:
        # drop some features (feature selection)
        df.drop(["n", "locCodeAndComment"], axis=1, inplace=True)

    # fill null values with 0
    df.fillna(0, inplace=True)
    df = df.replace(np.inf, 0)

    return df


def data_split(
    df: pd.DataFrame, split_size: float = 0.2, stratify_on_target: bool = True, target_sep: bool = True, save_dfs: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.

    Args:
        df (pd.DataFrame): Data to be split.
        split_size (float): train-test split ratio (test ratio).
        stratify_on_target (bool): Whether to do stratify split on target.
        target_sep (bool): Whether to do target setting for train and test sets.
        save_dfs (bool): Whether to save dataset splits in artifacts.

    Returns:
        train-test splits (with/without target setting)
    """
    if stratify_on_target:
        stra = df["defects"]
    else:
        stra = None

    train, test = train_test_split(df, test_size=split_size, random_state=42, stratify=stra)
    train = pd.DataFrame(train, columns=df.columns)
    test = pd.DataFrame(test, columns=df.columns)

    if save_dfs:
        df_obj = config.DataConfig()
        os.makedirs(os.path.dirname(df_obj.train_data_path), exist_ok=True)
        train.to_csv(df_obj.train_data_path)
        test.to_csv(df_obj.test_data_path)

    if target_sep:
        X_train, y_train = train.drop(["defects"], axis=1), train["defects"]
        X_test, y_test = test.drop(["defects"], axis=1), test["defects"]

        return X_train, X_test, y_train, y_test

    else:
        return train, test


def data_transformation(
    X_train: pd.DataFrame | np.ndarray, X_val: pd.DataFrame | np.ndarray = None, done_fe: bool = True
) -> Tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """Perform box-cox transform and standardization on data.

    Args:
        X_train : Training set (features)
        X_val : Validation/Test set (features)
        done_fe : Check if feature engineering is done as some features don't need box-cox transformation.

    Returns:
        Transformed X_train and X_val.
    """
    features = X_train.columns.tolist()
    if done_fe:
        lim = len(features) - 2
    else:
        lim = len(features)

    transformer = ColumnTransformer(
        [
            ("box_transform", BoxCoxTransformer(), features[:lim]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    scaler = Pipeline([("std_scaler", StandardScaler())])

    pipeline = Pipeline(steps=[("preprocessor", transformer), ("scaler", scaler)])

    pipeline.fit(X_train)
    utils.save_obj(scaler, config.artifacts_transformer_path)
    if type(X_val) == type(None):
        return X_train
    else:
        X_val = pipeline.transform(X_val)
        return X_train, X_val


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Box-cox transformer, custom sklearn transform."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform box-cox transformation on the data.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data.
        """
        for column in X.columns:
            X[column] = boxcox(X[column] + 1)[0]
        return X


if __name__ == "__main__":
    d_obj = config.DataConfig()
    df = load_dataset(d_obj.raw_data_path, print_info=True)
    df = clean_data(df)
    df = feature_engineering(df)
    X_train, X_val, y_train, y_val = data_split(df, save_dfs=True)
    X_train, X_val = data_transformation(X_train, X_val)
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
