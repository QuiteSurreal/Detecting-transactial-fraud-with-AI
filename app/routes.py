from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import json
from fastapi import FastAPI, BackgroundTasks, Request
from io import StringIO
import uuid

from app.services import preprocess as prep
from app.services import write as wr
from app.services import train

MODEL_PARAM_SCHEMAS = {
    "xgb": {
        "n_estimators": {"type": "number", "min": 100, "max": 2000},
        "max_depth": {"type": "number", "min": 3, "max": 15},
        "learning_rate": {"type": "number", "min": 0.001, "max": 0.5},
        "scale_pos_weight": {"type": "number", "min": 1, "max": 500},
        "eval_metric": {"type": "select", "options": ["logloss", "aucpr", "auc"]}
    },
    "xgb_smote": {
        "n_estimators": {"type": "number", "min": 100, "max": 2000},
        "max_depth": {"type": "number", "min": 3, "max": 15},
        "learning_rate": {"type": "number", "min": 0.001, "max": 0.5},
        "subsample": {"type": "number", "min": 0.1, "max": 1.0},
        "smote_sampling_strategy": {"type": "number", "min": 0.1, "max": 1.0},
        "eval_metric": {"type": "select", "options": ["logloss", "aucpr", "auc"]}
    },
    "ensemble": {
        "rf_n_estimators": {"type": "number", "min": 100, "max": 1000},
        "rf_max_depth": {"type": "number", "min": 3, "max": 15},
        "xgb_n_estimators": {"type": "number", "min": 100, "max": 2000},
        "xgb_max_depth": {"type": "number", "min": 3, "max": 15},
        "xgb_learning_rate": {"type": "number", "min": 0.001, "max": 0.5},
        "smote_sampling_strategy": {"type": "number", "min": 0.1, "max": 1.0}
    }
}



app = FastAPI()

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
def getStatsData(model_name: str = None):
    with open("app/utils/previous_data.json") as f:
        STATISTICS_ALL = json.load(f)

    if model_name:
        filtered = [item for item in STATISTICS_ALL if item.get("id") == model_name]
        if not filtered:
            raise HTTPException(status_code=404, detail=f"Statistics for model '{model_name}' not found")
        return filtered

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
async def predictFile(request: Request, background_tasks: BackgroundTasks, file: UploadFile, selected_model: str = Form(...)):
    task_id = str(uuid.uuid4())
    data = await file.read()
    entry = {
        "id": task_id,
        "status": "PENDING",
        "desc": "none"
    }
    wr.writeJSON(entry, "app/utils/tasks.json")
    background_tasks.add_task(runPreprocessFileJob, task_id, data, selected_model)
    
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/tasks", status_code=303)
    else:
        return {"status": "PENDING", "task_id": task_id}

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
async def trainModel(request: Request, background_tasks: BackgroundTasks, request_data: dict):

    errors = validateTrainData(request_data)

    if (errors):
        raise HTTPException(status_code=400, detail=errors)

    task_id = str(uuid.uuid4())
    entry = {
        "id": task_id,
        "status": "PENDING",
        "desc": "none"
    }

    wr.writeJSON(entry, "app/utils/tasks.json")
    background_tasks.add_task(runTrainJob, task_id, request_data, 0)
    
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/tasks", status_code=303)
    else:
        return {"status": "PENDING", "task_id": task_id}

@app.post("/upgrade")
async def upgrade(request: Request, background_tasks: BackgroundTasks, file: UploadFile, selected_model: str = Form(...)):
    
    with open("app/utils/model_registry.json") as f:
        MODEL_REGISTRY = json.load(f)
    
    model_info = MODEL_REGISTRY.get(selected_model)
    if not model_info or model_info.get("upgradable") != 1:
        raise HTTPException(status_code=400, detail="Model not found or not upgradable")
    
    task_id = str(uuid.uuid4())
    data = await file.read()
    entry = {
        "id": task_id,
        "status": "PENDING",
        "desc": "none"
    }
    wr.writeJSON(entry, "app/utils/tasks.json")
    
    request_data = {
        "base_model": model_info["base_model"],
        "model_path": model_info["path"],
        "model_name": selected_model + "_upgraded",
        "hyperparameters": {}
    }
    
    background_tasks.add_task(runTrainJob, task_id, request_data, 1, data)
    
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/tasks", status_code=303)
    else:
        return {"status": "PENDING", "task_id": task_id}

@app.post("/reset")
def resetSystem():
    with open("app/utils/tasks.json") as f:
        tasks = json.load(f)
    
    pending_tasks = [task for task in tasks if task.get("status") == "PENDING"]
    if pending_tasks:
        raise HTTPException(status_code=400, detail="Cannot reset system while tasks are pending")
    
    try:
        wr.reset()
        return {"message": "System reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")



def runPreprocessFileJob(task_id: str, file: bytes, selected_model: str):
    data = StringIO(file.decode("utf-8"))
    success, result, frauds, stats = prep.preprocessFile(data, selected_model)
    wr.writeTaskResult(task_id, success, result, frauds)
    if (stats):
        wr.writeStatsResult(selected_model, stats)

def runTrainJob(task_id: str, train_data: dict, mode: int, file_data: bytes = None):
    selected_model = train_data["base_model"]
    success, result, frauds, stats = train.prepareTrain(train_data, selected_model, mode, file_data)
    wr.writeTaskResult(task_id, success, result, frauds)
    if stats:
        wr.writeStatsResult(train_data["model_name"], stats)

def validateTrainData(request_data):
    errors = []
    
    if not isinstance(request_data.get("model_name"), str) or not request_data["model_name"].strip():
        errors.append("model_name must be a non-empty string")
    
    base_model = request_data.get("base_model")
    if base_model not in MODEL_PARAM_SCHEMAS:
        errors.append(f"base_model must be one of: {', '.join(MODEL_PARAM_SCHEMAS.keys())}")
        return errors  # Can't validate params if base_model invalid
    
    hyperparams = request_data.get("hyperparameters", {})
    if not isinstance(hyperparams, dict):
        errors.append("hyperparameters must be a dictionary")
        return errors
    
    schema = MODEL_PARAM_SCHEMAS[base_model]
    for param, rules in schema.items():
        if param not in hyperparams:
            errors.append(f"Missing required hyperparameter: {param}")
            continue
        
        value = hyperparams[param]
        if rules["type"] == "number":
            if not isinstance(value, (int, float)):
                errors.append(f"{param} must be a number")
            elif "min" in rules and value < rules["min"]:
                errors.append(f"{param} must be >= {rules['min']}")
            elif "max" in rules and value > rules["max"]:
                errors.append(f"{param} must be <= {rules['max']}")
        elif rules["type"] == "select":
            if value not in rules["options"]:
                errors.append(f"{param} must be one of: {', '.join(rules['options'])}")
    
    return errors
