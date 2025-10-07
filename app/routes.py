from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os, json
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from app.services import preprocess as prep

app = FastAPI()

tasks = {}

class PredictRequest(BaseModel):
    selected_model: str
    data: list

# Serve static files (CSS/JS) and templates
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

# Load model registry
with open("app/utils/model_registry.json") as f:
    MODEL_REGISTRY = json.load(f)


def load_model(selected_model: str):
    model_info = MODEL_REGISTRY.get(selected_model)
    if not model_info:
        raise ValueError(f"Model {selected_model} not found")
    return joblib.load(model_info["path"])

@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
def index():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

    
@app.get("/models")
def get_models():
    with open("app/utils/model_registry.json") as f:
        MODEL_REGISTRY = json.load(f)
    return MODEL_REGISTRY

@app.post("/upgrade", response_class=HTMLResponse)
def wo():
    with open("templates/stats.html") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict/file")
def predictFile(background_tasks: BackgroundTasks, file: UploadFile, selected_model: str = Form(...)):
    import uuid
    task_id = str(uuid.uuid4())
    status = "PENDING"
    tasks[task_id] = {"status": status, "result": None}
    entry = {
        "id": task_id,
        "status": status,
        "desc": "none"
    }
    writeJSON(entry, "app/utils/tasks.json")
    background_tasks.add_task(runPreprocessFileJob, task_id, file, selected_model, 1)
    
    return {"task_id": task_id}

@app.post("/predict/json")
def predictJSON(background_tasks: BackgroundTasks, request: PredictRequest):
    import uuid
    task_id = str(uuid.uuid4())
    status = "PENDING"
    tasks[task_id] = {"status": status, "result": None}
    entry = {
        "id": task_id,
        "status": status,
        "desc": "none"
    }
    writeJSON(entry, "app/utils/tasks.json")
    background_tasks.add_task(runPreprocessJSONJob, task_id, request, 1)
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    return tasks.get(task_id, {"status": "UNKNOWN"})

def runPreprocessFileJob(task_id: str, file: UploadFile, selected_model: str, mode: int):
    result = prep.preprocessFile(file, selected_model, mode)
    writeTaskResult(task_id, result)

def runPreprocessJSONJob(task_id: str, request: PredictRequest, mode: int):
    result = prep.preprocessJSON(request, mode)
    writeTaskResult(task_id, result)

def writeTaskResult(task_id, result):
    if True:
        tasks[task_id]['status'] = "SUCCESS"
        tasks[task_id]['desc'] = result

        entry = {
        "id": task_id,
        "status": "SUCCESS",
        "desc": result
        }
        editJSON(task_id, entry, "app/utils/tasks.json")
        print(entry)

def writeJSON(new_data, filename):
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data.append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

#replace at the same pos
def editJSON(id, new_data, filename):
    with open(filename, 'r+') as file:
        file_data = json.load(file)

        for i, obj in enumerate(file_data):
            if obj['id'] == id:
                file_data[i] = new_data
                break
            
        file.seek(0)
        file.truncate()
        json.dump(file_data, file, indent=4)

# Purge tasks.json, only for the dev time
with open("app/utils/tasks.json", "w") as f:
    json.dump([], f)