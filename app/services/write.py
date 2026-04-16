import json
import numpy as np
import math
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def writeTaskResult(task_id, success, result, frauds):
    if success:
        entry = {
        "id": task_id,
        "status": "SUCCESS",
        "desc": result,
        "frauds": frauds
        }

    else:
        entry = {
        "id": task_id,
        "status": "FAIL",
        "desc": result
        }
    
    editJSON(task_id, entry, "app/utils/tasks.json")

def writeTrainResult(task_id, success, result):
    if success:
        entry = {
        "id": task_id,
        "status": "SUCCESS",
        "desc": result,
        }

    else:
        entry = {
        "id": task_id,
        "status": "FAIL",
        "desc": result
        }
    
    editJSON(task_id, entry, "app/utils/tasks.json")

def writeStatsResult(model_name, stats):
    with open("app/utils/previous_data.json") as f:
        prev = json.load(f)

        conf = [
            [
                stats[3][0][0] + prev[0]["confusion"][0][0],
                stats[3][0][1] + prev[0]["confusion"][0][1]
            ],
            [
                stats[3][1][0] + prev[0]["confusion"][1][0],
                stats[3][1][1] + prev[0]["confusion"][1][1]
            ]
        ]
        try:
            t = threading.Thread(target=makeCM, args=(conf, "Total"), daemon=True)
            t.start()
        except Exception:
            path = makeCM(conf, "Total")
        wg = [stats[0], prev[0]["records"]]
        entry = {
            "id": "Total",
            "records": wg[0] + wg[1],
            "frauds": stats[1] + prev[0]["frauds"],
            "legit": stats[2] + prev[0]["legit"],
            "confusion": conf,
            "acc": math.floor(np.average([stats[4], prev[0]["acc"]], weights=wg) * 1000) / 1000,
            "prec": math.floor(np.average([stats[5], prev[0]["prec"]], weights=wg) * 1000) / 1000,
            "rec": math.floor(np.average([stats[6], prev[0]["rec"]], weights=wg) * 1000) / 1000,
            "F1": math.floor(np.average([stats[7], prev[0]["F1"]], weights=wg) * 1000) / 1000,
        }
        editJSON("Total", entry, "app/utils/previous_data.json")
    
        for i, obj in enumerate(prev):
            if obj['id'] == model_name:
                conf = [
                    [
                        stats[3][0][0] + prev[i]["confusion"][0][0],
                        stats[3][0][1] + prev[i]["confusion"][0][1]
                    ],
                    [
                        stats[3][1][0] + prev[i]["confusion"][1][0],
                        stats[3][1][1] + prev[i]["confusion"][1][1]
                    ]
                ]
                try:
                    t = threading.Thread(target=makeCM, args=(conf, model_name), daemon=True)
                    t.start()
                except Exception:
                    path = makeCM(conf, model_name)
                wg = [stats[0], prev[i]["records"]]
                entry = {
                "id": model_name,
                "records": wg[0] + wg[1],
                "frauds": stats[1] + prev[i]["frauds"],
                "legit": stats[2] + prev[i]["legit"],
                "confusion": conf,
                "acc": math.floor(np.average([stats[4], prev[i]["acc"]], weights=wg) * 1000) / 1000,
                "prec": math.floor(np.average([stats[5], prev[i]["prec"]], weights=wg) * 1000) / 1000,
                "rec": math.floor(np.average([stats[6], prev[i]["rec"]], weights=wg) * 1000) / 1000,
                "F1": math.floor(np.average([stats[7], prev[i]["F1"]], weights=wg) * 1000) / 1000
                }
                editJSON(model_name, entry, "app/utils/previous_data.json")
                break
        else:
            # Model not found, create new entry
            try:
                t = threading.Thread(target=makeCM, args=(stats[3], model_name), daemon=True)
                t.start()
            except Exception:
                path = makeCM(stats[3], model_name)
            entry = {
                "id": model_name,
                "records": stats[0],
                "frauds": stats[1],
                "legit": stats[2],
                "confusion": stats[3],
                "acc": math.floor(stats[4] * 1000) / 1000,
                "prec": math.floor(stats[5] * 1000) / 1000,
                "rec": math.floor(stats[6] * 1000) / 1000,
                "F1": math.floor(stats[7] * 1000) / 1000
            }
            writeJSON(entry, "app/utils/previous_data.json")
        

def makeCM(conf, name):

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    output_path = f"./resources/temp/cm_{name}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


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

def updateModelRegistry(model_name, model_path, description, base_model=None, hyperparameters=None):
    

    # Load existing registry
    with open("app/utils/model_registry.json", 'r') as f:
        registry = json.load(f)

    # Add new model
    registry[model_name] = {
        "path": model_path,
        "description": description,
        "base_model": base_model,
        "hyperparameters": hyperparameters
    }

    # Save updated registry
    with open("app/utils/model_registry.json", 'w') as f:
        json.dump(registry, f, indent=2)