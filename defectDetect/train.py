import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from catboost import CatBoostClassifier
from defectDetect import data
from defectDetect.config import (DataConfig, ModelConfig,
                                 artifacts_model_param_path,
                                 artifacts_results_path,
                                 artifacts_tuned_model_param_path, logger)
from defectDetect.utils import read_yaml, save_models, write_yaml
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def hill_climb_ensemble(val_pred_df: pd.DataFrame, y_val: pd.Series, weight_range):
    logger.info("Initialzing hill climb ensemble.")
    STOP = False
    current_best_ensemble = val_pred_df.iloc[:, 0]
    MODELS = val_pred_df.iloc[:, 1:]
    weight_range = weight_range
    model_weights = {}
    history = {val_pred_df.columns[0]: float(roc_auc_score(y_val, current_best_ensemble))}
    i = 0

    # Hill climbing
    while not STOP:
        i += 1
        potential_new_best_cv_score = roc_auc_score(y_val, current_best_ensemble)
        k_best, wgt_best = None, None
        for k in MODELS:
            for wgt in weight_range:
                potential_ensemble = (1 - wgt) * current_best_ensemble + wgt * MODELS[k]
                cv_score = roc_auc_score(y_val, potential_ensemble)
                if cv_score > potential_new_best_cv_score:
                    potential_new_best_cv_score = cv_score
                    k_best, wgt_best = k, wgt

        if k_best is not None:
            current_best_ensemble = (1 - wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
            MODELS.drop(k_best, axis=1, inplace=True)
            if MODELS.shape[1] == 0:
                STOP = True
            print(f"Iteration: {i}, Model added: {k_best}, Best weight: {wgt_best:.2f}, Best AUC: {potential_new_best_cv_score:.5f}")
            model_weights[k_best] = float(wgt_best)
            history["".join(list(history.keys())[-1]) + " + " + k_best] = float(potential_new_best_cv_score)
        else:
            STOP = True

    return model_weights, history, current_best_ensemble


def voting_ensemble(models, X_train, y_train, X_val, voting):
    logger.info("Initializing hill climb ensemble.")
    votclf = VotingClassifier(
        estimators=models,
        voting=voting,
    )
    votclf.fit(X_train, y_train)
    val_votclf_preds = votclf.predict_proba(X_val)[:, 1]

    return votclf, val_votclf_preds


def train(weight_range, voting, params_path=None):
    d_obj = DataConfig()
    df = data.load_dataset(d_obj.raw_data_path)
    df = data.clean_data(df)
    df = data.feature_engineering(df)
    X_train, X_val, y_train, y_val = data.data_split(df, save_dfs=True)
    X_train, X_val = data.data_transformation(X_train, X_val)

    logger.info("Training models.")
    logreg_model = LogisticRegression(random_state=1234)
    knnclf_model = KNeighborsClassifier()
    rfclf_model = RandomForestClassifier(random_state=1234)
    lgbmclf_model = LGBMClassifier(random_state=1234, verbose=-1)
    xgbclf_model = XGBClassifier(random_state=1234)
    cbclf_model = CatBoostClassifier(random_state=1234, verbose=0)

    if type(params_path) != type(None):
        params = read_yaml(params_path)
        try:
            logreg_model.set_params(**params["LogisticRegression"])
        except:
            pass

        try:
            knnclf_model.set_params(**params["KNeighborsClassifier"])
        except:
            pass

        try:
            rfclf_model.set_params(**params["RandomForestClassifier"])
        except:
            pass

        try:
            lgbmclf_model.set_params(**params["LGBMClassifier"])
        except:
            pass

        try:
            xgbclf_model.set_params(**params["XGBClassifier"])
        except:
            pass

        try:
            cbclf_model.set_params(**params["CatBoostClassifier"])
        except:
            pass

    logreg_model.fit(X_train, y_train)
    knnclf_model.fit(X_train, y_train)
    rfclf_model.fit(X_train, y_train)
    lgbmclf_model.fit(X_train, y_train)
    xgbclf_model.fit(X_train, y_train)
    cbclf_model.fit(X_train, y_train)

    artifacts = {}
    results = {}

    artifacts["LogisticRegression"] = logreg_model.get_params()
    artifacts["KNeighborsClassifier"] = knnclf_model.get_params()
    artifacts["RandomForestClassifier"] = rfclf_model.get_params()
    artifacts["LGBMClassifier"] = lgbmclf_model.get_params()
    artifacts["XGBClassifier"] = xgbclf_model.get_params()
    artifacts["CatBoostClassifier"] = cbclf_model.get_params()

    val_preds = {}

    val_preds["LogisticRegression"] = logreg_model.predict_proba(X_val)[:, 1]
    val_preds["KNeighborsClassifier"] = knnclf_model.predict_proba(X_val)[:, 1]
    val_preds["RandomForestClassifier"] = rfclf_model.predict_proba(X_val)[:, 1]
    val_preds["LGBMClassifier"] = lgbmclf_model.predict_proba(X_val)[:, 1]
    val_preds["XGBClassifier"] = xgbclf_model.predict_proba(X_val)[:, 1]
    val_preds["CatBoostClassifier"] = cbclf_model.predict_proba(X_val)[:, 1]

    val_pred_df = pd.DataFrame(val_preds)

    model_weights, history, val_hill_climb_preds = hill_climb_ensemble(val_pred_df, y_val, weight_range)

    artifacts["hillclimb_weights"] = model_weights
    results["hillclimb"] = history
    models = [
        ("logreg", logreg_model),
        ("knnclf", knnclf_model),
        ("rfclf", rfclf_model),
        ("lgbmclf", lgbmclf_model),
        ("xgbclf", xgbclf_model),
        ("cbclf", cbclf_model),
    ]

    votclf_model, val_votclf_preds = voting_ensemble(models, X_train, y_train, X_val, voting)
    results["votclf"] = float(roc_auc_score(y_val, val_votclf_preds))

    final_val_preds = (val_hill_climb_preds + val_votclf_preds) / 2
    score = float(roc_auc_score(y_val, final_val_preds))
    print(score)
    results["complete"] = score

    write_yaml(artifacts, artifacts_model_param_path)
    write_yaml(results, artifacts_results_path)

    models.append(("votclf", votclf_model))
    names, act_models = zip(*models)
    save_models(act_models, names)


if __name__ == "__main__":
    model_cfg = ModelConfig()
    train(model_cfg.weight_range, model_cfg.voting, artifacts_tuned_model_param_path)
