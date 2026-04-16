import joblib
import os
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
    
    # Predict anomalies with Isolation Forest (-1 for anomalies, 1 for normal)
    anomaly_scores = isolation_forest.predict(df)
    anomalies = anomaly_scores == -1
    
    # For anomalies, predict clusters with KMeans
    if anomalies.sum() > 0:
        cluster_labels = kmeans.predict(df[anomalies])
    else:
        cluster_labels = []
    
    # Return a list or dict with anomaly status and cluster
    results = []
    cluster_idx = 0
    for i, is_anomaly in enumerate(anomalies):
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
    return joblib.load(model_info["path"])