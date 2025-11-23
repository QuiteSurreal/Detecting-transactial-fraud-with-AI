import joblib
import os
import pandas as pd
import json

MODEL_PATH = "app/models"

def runPrediction(modelName, df: pd.DataFrame):
    model = loadModel(modelName)

    result = model.predict(df)
    return result

def loadModel(modelName):
    with open("app/utils/model_registry.json") as f:
        registry = json.load(f)
        model_info = registry[modelName]
    if not model_info:
        raise ValueError(f"Model {modelName} not found")
    return joblib.load(model_info["path"])