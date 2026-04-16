from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import json
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from io import StringIO
import uuid

from app.services import preprocess as prep
from app.services import write as wr
from app.services import train

app = FastAPI()

class PredictRequest(BaseModel):
    selected_model: str
    data: list

# Serve static files (CSS/JS) and templates
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

# Load model registry
with open("app/utils/model_registry.json") as f:
    MODEL_REGISTRY = json.load(f)


@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
def index():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())
    
@app.get("/tasks", response_class=HTMLResponse)
def getTasksPage():
    with open("templates/tasks.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/taskDetails", response_class=HTMLResponse)
def getTaskDetails():
    with open("templates/taskDetails.html") as f:
        return HTMLResponse(content=f.read())
    
@app.get("/stats", response_class=HTMLResponse)
def getStats():
    with open("templates/stats.html") as f:
        return HTMLResponse(content=f.read())
    
@app.get("/train", response_class=HTMLResponse)
def getTrain():
    with open("templates/train.html") as f:
        return HTMLResponse(content=f.read())
    
@app.get("/statsData")
def getStatsData():
    with open("app/utils/previous_data.json") as f:
        STATISTICS_ALL = json.load(f)
    return STATISTICS_ALL
    
@app.get("/models")
def getModels():
    with open("app/utils/model_registry.json") as f:
        MODEL_REGISTRY = json.load(f)
    return MODEL_REGISTRY

@app.get("/tasks/data")
def getTasksData(task_id: str = None):
    with open("app/utils/tasks.json") as f:
        DATA = json.load(f)
    
    if task_id:
        for task in DATA:
            if task['id'] == task_id:
                return task
        raise HTTPException(status_code=404, detail="Task not found")
    return DATA

@app.post("/upgrade", response_class=HTMLResponse)
def wo():
    with open("templates/stats.html") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict/file")
async def predictFile(background_tasks: BackgroundTasks, file: UploadFile, selected_model: str = Form(...)):
    task_id = str(uuid.uuid4())
    data = await file.read()
    entry = {
        "id": task_id,
        "status": "PENDING",
        "desc": "none"
    }
    wr.writeJSON(entry, "app/utils/tasks.json")
    background_tasks.add_task(runPreprocessFileJob, task_id, data, selected_model, 1)
    
    return RedirectResponse(url="/", status_code=303)

@app.post("/predict/json")
def predictJSON(background_tasks: BackgroundTasks, request: PredictRequest):
    task_id = str(uuid.uuid4())
    entry = {
        "id": task_id,
        "status": "PENDING",
        "desc": "none"
    }
    wr.writeJSON(entry, "app/utils/tasks.json")
    background_tasks.add_task(runPreprocessJSONJob, task_id, request, 1)
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    with open("app/utils/tasks.json") as f:
        DATA = json.load(f)
    
    if task_id:
        for task in DATA:
            if task['id'] == task_id:
                return task['status']
        raise HTTPException(status_code=404, detail="Task not found")
    return DATA

@app.post("/train")
async def trainModel(background_tasks: BackgroundTasks, request_data: dict):
    task_id = str(uuid.uuid4())
    entry = {
        "id": task_id,
        "status": "PENDING",
        "desc": "none"
    }
    wr.writeJSON(entry, "app/utils/tasks.json")
    background_tasks.add_task(runTrainJob, task_id, request_data)
    
    return RedirectResponse(url="/", status_code=303)
    #return {"task_id": task_id}





def runPreprocessFileJob(task_id: str, file: bytes, selected_model: str, mode: int):
    data = StringIO(file.decode("utf-8"))
    success, result, frauds, stats = prep.preprocessFile(data, selected_model, mode)
    wr.writeTaskResult(task_id, success, result, frauds)
    wr.writeStatsResult(selected_model, stats)

def runPreprocessJSONJob(task_id: str, request: PredictRequest, mode: int):
    success, result = prep.preprocessJSON(request, mode)
    wr.writeTaskResult(task_id, success, result, [])

def runTrainJob(task_id: str, train_data: dict):
    selected_model = train_data["base_model"]
    success, result, frauds, stats = train.prepareTrain(train_data, selected_model)
    wr.writeTaskResult(task_id, success, result, frauds)
    if stats:
        wr.writeStatsResult(train_data["model_name"], stats)



#Purge tasks.json, only for the dev time
with open("app/utils/tasks.json", "w") as f:
    json.dump([], f)