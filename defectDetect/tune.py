import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier

import optuna
from catboost import CatBoostClassifier
from defectDetect import data
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# Define an objective function to optimize
def objective(trial, model_name):
    # Define hyperparameters to optimize
    models = {
        "LogisticRegression": LogisticRegression(random_state=1234),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RandomForestClassifier": RandomForestClassifier(random_state=1234),
        "LGBMClassifier": LGBMClassifier(random_state=1234, verbose=-1),
        "XGBClassifier": XGBClassifier(random_state=1234),
        "CatBoostClassifier": CatBoostClassifier(random_state=1234, verbose=0),
    }

    params = {
        "LogisticRegression": dict(
            C=trial.suggest_float("C", 0.7, 1),
            solver=trial.suggest_categorical("solver", ["liblinear", "newton-cholesky", "saga"]),
            max_iter=trial.suggest_int("max_iter", 50, 150, step=50),
        ),
        "KNeighborsClassifier": dict(
            n_neighbors=trial.suggest_int("n_neighbors", 50, 500, 50),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        ),
        "RandomForestClassifier": dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 8, step=2),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 5),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
        ),
        "LGBMClassifier": dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 8, step=2),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 0.3),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 0.2),
        ),
        "XGBClassifier": dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 8, step=2),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 0.5),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 0.3),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 0.2),
        ),
        "CatBoostClassifier": dict(
            iterations=trial.suggest_int("iterations", 50, 500, step=50),
            depth=trial.suggest_int("depth", 2, 8, step=2),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 2, 32, log=True),
        ),
    }

    # Create a logistic regression with suggested hyperparameters
    model = models[model_name]
    model.set_params(**params[model_name])

    # Perform stratified k-fold cross-validation with train-test splits
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Adjust n_splits and other parameters as needed
    scores = []

    df = data.load_dataset(r"datasets\raw\jm1.csv")
    df = data.clean_data(df)
    df = data.feature_engineering(df)
    df_X = df.drop(["defects"], axis=1)
    X, y = data.data_transformation(df_X), df["defects"]
    X = pd.DataFrame(X, columns=df_X.columns)
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        scores.append(score)

    # Calculate the mean accuracy
    mean_roc_auc_score = np.mean(scores)

    return mean_roc_auc_score


def tune():
    models = [
        "LogisticRegression",
        "KNeighborsClassifier",
        "RandomForestClassifier",
        "LGBMClassifier",
        "XGBClassifier",
        "CatBoostClassifier",
    ]
    artifacts = {}

    for model_name in models:
        # Create an Optuna study
        study = optuna.create_study(direction="maximize")

        # Optimize the objective function
        study.optimize(lambda trial: objective(trial, model_name), n_trials=25)

        # Print the best hyperparameters and their corresponding loss (1 - accuracy)
        best_params = study.best_params
        best_roc_auc_score = study.best_value

        print("Best Hyperparameters:", best_params)
        print("Best Accuracy:", best_roc_auc_score)

        artifacts[model_name] = {"hyperparameters": best_params, "roc_auc_score": best_roc_auc_score}

    return artifacts


if __name__ == "__main__":
    artifacts = tune()
    print(artifacts)
