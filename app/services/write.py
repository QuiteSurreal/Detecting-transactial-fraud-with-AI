import json
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import portalocker

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
            makeCM(conf, "Total")
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
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
                    makeCM(conf, model_name)
                except Exception as e:
                    print(f"Error generating confusion matrix: {e}")
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
            # model not found, create new entry
            try:
                makeCM(stats[3], model_name)
            except Exception as e:
                print(f"Error generating confusion matrix: {e}")
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


def writeJSON(new_data, filename):
    with open(filename, 'r+') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file_data = json.load(file)
        file_data.append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

#replace at the same pos
def editJSON(id, new_data, filename):
    with open(filename, 'r+') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file_data = json.load(file)

        for i, obj in enumerate(file_data):
            if obj['id'] == id:
                file_data[i] = new_data
                break
            
        file.seek(0)
        file.truncate()
        json.dump(file_data, file, indent=4)

def updateModelRegistry(model_name, model_path, description, base_model, upgradable, hyperparameters=None):
    
    with open("app/utils/model_registry.json", 'r') as f:
        registry = json.load(f)

    registry[model_name] = {
        "path": model_path,
        "description": description,
        "base_model": base_model,
        "upgradable": upgradable,
        "hyperparameters": hyperparameters
    }

    with open("app/utils/model_registry.json", 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(registry, f, indent=2)

def reset():
    import os

    #reset JSON files
    with open("app/utils/tasks.json", "w") as f:
        json.dump([], f, indent=4)
    with open("app/utils/previous_data_reset.json", "r") as r:
        reset = json.load(r)
    with open("app/utils/previous_data.json", "w") as f:
        json.dump(reset, f, indent=4)
    with open("app/utils/model_registry_reset.json", "r") as r:
        reset = json.load(r)
    with open("app/utils/model_registry.json", "w") as f:
        json.dump(reset, f, indent=4)

    #load protected files list
    protected_files = set()
    try:
        with open("app/utils/protected_files.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    protected_files.add(line)
    except FileNotFoundError:
        print("protected_files.txt not found, skipping file cleanup")

    #clean up generated model files
    models_dir = "app/models/"
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            filepath = os.path.join(models_dir, filename)
            if filepath not in protected_files and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                    print(f"Removed generated model file: {filepath}")
                except Exception as e:
                    print(f"Error removing {filepath}: {e}")

    #clean up generated confusion matrix images
    temp_dir = "resources/temp/"
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            if filename.startswith("cm_") and filename.endswith(".png"):
                filepath = os.path.join(temp_dir, filename)
                try:
                    os.remove(filepath)
                    print(f"Removed generated confusion matrix: {filepath}")
                except Exception as e:
                    print(f"Error removing {filepath}: {e}")

    print("Reset completed successfully!")