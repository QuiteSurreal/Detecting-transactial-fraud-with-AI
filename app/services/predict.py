import joblib
import xgboost as xgb
import pandas as pd
import json
import shap

MODEL_PATH = "app/models"

def runPrediction(model_name, df: pd.DataFrame):
    """
    Routes prediction to supervised or unsupervised model and returns predictions with feature importance.
    
    Args:
        model_name (str): Registered model name to use for prediction
        df (pd.DataFrame): Preprocessed features
    
    Returns:
        tuple: (predictions, explanations) - fraud predictions and SHAP feature importance dict
    """
    if model_name == "Isolation Forest + KMeans":
        return runUnsupervisedPrediction(df)
    else:
        model, is_ens = loadModel(model_name)
        result = model.predict(df)
        if (is_ens):
            model = model = model.estimators_[1]
        exp = makeExplanation(model, df)
        return result, exp

def runUnsupervisedPrediction(df: pd.DataFrame):
    """
    Uses Isolation Forest for anomaly detection and KMeans for clustering anomalies.
    
    Args:
        df (pd.DataFrame): Preprocessed features
    
    Returns:
        tuple: (results, None) - list of dicts with is_anomaly and cluster assignment, no explanations
    """
    with open("app/utils/model_registry.json") as f:
        registry = json.load(f)
        model_info = registry["Isolation Forest + KMeans"]
    
    isolation_forest = joblib.load(model_info["pathI"])
    kmeans = joblib.load(model_info["pathK"])
    
    anomaly_scores = isolation_forest.predict(df)
    
    results = []

    for i in range(len(df)):

        if anomaly_scores[i] == -1:
            cluster_id = kmeans.predict(df.iloc[[i]])[0]

            results.append({
                'is_anomaly': True,
                'cluster': int(cluster_id)
            })
        else:
            results.append({
                'is_anomaly': False,
                'cluster': None
            })
    
    return results, None

def makeExplanation(model, df):
    """
    Computes SHAP-based feature importance
    
    Args:
        model: Trained supervised model (XGBoost or ensemble)
        df (pd.DataFrame): Preprocessed features
    
    Returns:
        dict: Feature name: SHAP importance value, sorted descending by absolute importance
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    features = df.columns

    importance = dict(zip(features, shap_values[0].tolist()))

    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))


def loadModel(model_name):
    """
    Loads a registered model from disk and returns model object with ensemble flag.
    
    Args:
        model_name (str): Registered model name from model_registry.json
    
    Returns:
        tuple: (model_object, is_ensemble_flag) - loaded model and 1 if ensemble else 0
    """
    with open("app/utils/model_registry.json") as f:
        registry = json.load(f)
        model_info = registry[model_name]
    if not model_info:
        raise ValueError(f"Model {model_name} not found")
    if (model_info["base_model"] == "xgb" or model_info["base_model"] == "xgb_smote"):
        model = xgb.XGBClassifier()
        model.load_model(model_info["path"])
        return model, 0
    else:
        return joblib.load(model_info["path"]), 1