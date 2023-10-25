import os

import joblib

from defectDetect import config, data, utils


def prediction(df):
    bxcx = data.BoxCoxTransformer()
    transformer = joblib.load(config.artifacts_transformer_path)

    df = data.clean_data(df)
    X = data.feature_engineering(df)
    try:
        X = bxcx.transform(X)
    except:
        pass
    X = transformer.transform(X)

    logregclf_model = joblib.load(os.path.join(config.artifacts_model_pickle_path, "logreg.pkl"))
    knnclf_model = joblib.load(os.path.join(config.artifacts_model_pickle_path, "knnclf.pkl"))
    rfclf_model = joblib.load(os.path.join(config.artifacts_model_pickle_path, "rfclf.pkl"))
    lgbmclf_model = joblib.load(os.path.join(config.artifacts_model_pickle_path, "lgbmclf.pkl"))
    xgbclf_model = joblib.load(os.path.join(config.artifacts_model_pickle_path, "xgbclf.pkl"))
    cbclf_model = joblib.load(os.path.join(config.artifacts_model_pickle_path, "cbclf.pkl"))
    votclf_model = joblib.load(os.path.join(config.artifacts_model_pickle_path, "votclf.pkl"))

    params = utils.read_yaml(config.artifacts_model_param_path)

    y_pred_hc = logregclf_model.predict_proba(X)[:, 1]
    for k, v in params["hillclimb_weights"].items():
        if k == "KNeighborsClassifier":
            y_pred_hc = y_pred_hc * (1 - v) + knnclf_model.predict_proba(X)[:, 1] * v
        elif k == "RandomForestClassifier":
            y_pred_hc = y_pred_hc * (1 - v) + rfclf_model.predict_proba(X)[:, 1] * v
        elif k == "LGBMClassifier":
            y_pred_hc = y_pred_hc * (1 - v) + lgbmclf_model.predict_proba(X)[:, 1] * v
        elif k == "XGBClassifier":
            y_pred_hc = y_pred_hc * (1 - v) + xgbclf_model.predict_proba(X)[:, 1] * v
        elif k == "CatBoostClassifier":
            y_pred_hc = y_pred_hc * (1 - v) + cbclf_model.predict_proba(X)[:, 1] * v

    y_pred_vc = votclf_model.predict_proba(X)[:, 1]

    y_pred = (y_pred_hc + y_pred_vc) / 2

    return y_pred[0]
