from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
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

app.mount("/resources", StaticFiles(directory="resources"), name="resources")


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
    background_tasks.add_task(runPreprocessFileJob, task_id, data, selected_model)
    
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
    background_tasks.add_task(runPreprocessJSONJob, task_id, request)
    
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
    background_tasks.add_task(runTrainJob, task_id, request_data, 0)
    
    return RedirectResponse(url="/", status_code=303)
    #return {"task_id": task_id}

@app.post("/upgrade")
async def upgrade(background_tasks: BackgroundTasks, file: UploadFile, selected_model: str = Form(...)):
    task_id = str(uuid.uuid4())
    data = await file.read()
    entry = {
        "id": task_id,
        "status": "PENDING",
        "desc": "none"
    }
    wr.writeJSON(entry, "app/utils/tasks.json")

    with open("app/utils/model_registry.json") as f:
        MODEL_REGISTRY = json.load(f)
    
    model_info = MODEL_REGISTRY.get(selected_model)
    if not model_info or model_info.get("upgradable") != 1:
        raise HTTPException(status_code=400, detail="Model not found or not upgradable")
    
    request_data = {
        "base_model": model_info["base_model"],
        "model_path": model_info["path"],
        "model_name": selected_model + "_upgraded",
        "hyperparameters": {}
    }
    
    background_tasks.add_task(runTrainJob, task_id, request_data, 1, data)
    
    return RedirectResponse(url="/", status_code=303)





def runPreprocessFileJob(task_id: str, file: bytes, selected_model: str):
    data = StringIO(file.decode("utf-8"))
    success, result, frauds, stats = prep.preprocessFile(data, selected_model)
    wr.writeTaskResult(task_id, success, result, frauds)
    if (stats):
        wr.writeStatsResult(selected_model, stats)

def runPreprocessJSONJob(task_id: str, request: PredictRequest):
    success, result = prep.preprocessJSON(request)
    wr.writeTaskResult(task_id, success, result, [])

def runTrainJob(task_id: str, train_data: dict, mode: int, file_data: bytes = None):
    selected_model = train_data["base_model"]
    success, result, frauds, stats = train.prepareTrain(train_data, selected_model, mode, file_data)
    wr.writeTaskResult(task_id, success, result, frauds)
    if stats:
        wr.writeStatsResult(train_data["model_name"], stats)

#Purge tasks.json, only for the dev time
with open("app/utils/tasks.json", "w") as f:
    json.dump([], f)