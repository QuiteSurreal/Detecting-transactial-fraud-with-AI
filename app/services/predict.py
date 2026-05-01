import joblib
import xgboost as xgb
import pandas as pd
import json
import shap

MODEL_PATH = "app/models"

def runPrediction(model_name, df: pd.DataFrame):
    if model_name == "Isolation Forest + KMeans":
        return runUnsupervisedPrediction(df)
    else:
        model, is_ens = loadModel(model_name)
        y_scores = model.predict_proba(df)
        result = (y_scores[:, 1] >= 0.8105).astype(int)
        if (is_ens):
            model = model = model.estimators_[1]
        exp = makeExplanation(model, df)
        return result, exp

def runUnsupervisedPrediction(df: pd.DataFrame):
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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    features = df.columns

    importance = dict(zip(features, shap_values[0].tolist()))

    print(importance)

    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))


def loadModel(model_name):
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