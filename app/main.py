from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os, json

app = FastAPI()

# Serve static files (CSS/JS) and templates
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model registry
with open("app/utils/model_registry.json") as f:
    MODEL_REGISTRY = json.load(f)

def load_model(model_name: str):
    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found")
    return joblib.load(model_info["path"])

@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile, model_name: str = Form(...)):
    # Read uploaded CSV
    df = pd.read_csv(file.file)

    # Load selected model
    model = load_model(model_name)

    # Preprocess (dummy example)
    X = df.drop(columns=["label"], errors="ignore")  # ignore label if missing

    # Predict
    y_pred = model.predict(X)

    # Return results
    df["prediction"] = y_pred
    results_path = f"app/data/results_{file.filename}"
    df.to_csv(results_path, index=False)

    return JSONResponse({
        "message": "Prediction completed",
        "model": model_name,
        "output_file": results_path
    })