import joblib
import xgboost as xgb
import pandas as pd
import json

MODEL_PATH = "app/models"

def runPrediction(modelName, df: pd.DataFrame):
    if modelName == "Isolation Forest + KMeans":
        return runUnsupervisedPrediction(df)
    else:
        model = loadModel(modelName)
        result = model.predict(df)
        return result

def runUnsupervisedPrediction(df: pd.DataFrame):
    with open("app/utils/model_registry.json") as f:
        registry = json.load(f)
        model_info = registry["Isolation Forest + KMeans"]
    
    isolation_forest = joblib.load(model_info["pathI"])
    kmeans = joblib.load(model_info["pathK"])
    
    anomaly_scores = isolation_forest.predict(df)
    anomalies = anomaly_scores == -1
    
    if anomalies.sum() > 0:
        cluster_labels = kmeans.predict(df[anomalies])
    else:
        cluster_labels = []
    
    results = []
    cluster_idx = 0
    for is_anomaly in enumerate(anomalies):
        if is_anomaly:
            results.append({
                'is_anomaly': True,
                'cluster': int(cluster_labels[cluster_idx])
            })
            cluster_idx += 1
        else:
            results.append({
                'is_anomaly': False,
                'cluster': None
            })
    
    return results

def loadModel(modelName):
    with open("app/utils/model_registry.json") as f:
        registry = json.load(f)
        model_info = registry[modelName]
    if not model_info:
        raise ValueError(f"Model {modelName} not found")
    if (model_info["base_model"] == "xgb" or model_info["base_model"] == "xgb_smote"):
        model = xgb.XGBClassifier()
        model.load_model(model_info["path"])
        return model
    else:
        return joblib.load(model_info["path"])