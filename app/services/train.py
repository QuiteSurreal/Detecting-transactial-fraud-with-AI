from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

from imblearn.over_sampling import SMOTE

import joblib

from app.services import write as wr


X_train = None
y_train = None
X_test = None
y_test = None





def prepareTrain(train_data, selected_model):
    df = pd.read_csv('./factory/Data/preprocessed_input.csv', delimiter=',')

    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 42)

    model_name = train_data["model_name"]

    if (selected_model == "xgb"):
        return trainXGB(train_data)
    elif (selected_model == "xgb_smote"):
        return trainXGBSMOTE(train_data)
    elif (selected_model == "ensemble"):
        return trainEnsemble(train_data)


def trainXGB(train_data):
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    model = xgb.XGBClassifier(
        n_estimators=train_data["hyperparameters"]["n_estimators"],
        max_depth=train_data["hyperparameters"]["max_depth"],
        learning_rate=train_data["hyperparameters"]["learning_rate"],
        scale_pos_weight=train_data["hyperparameters"]["scale_pos_weight"],
        tree_method='hist',
        eval_metric=train_data["hyperparameters"]["eval_metric"]
    )

    model.fit(X_train, y_train)

    model_path = f"app/models/{train_data['model_name']}.sav"
    joblib.dump(model, model_path)

    wr.updateModelRegistry(train_data['model_name'], model_path, "Custom XGB model")

    y_pred = model.predict(X_test)
    fraud_count = int((y_pred == 1).sum())
    legit_count = int((y_pred == 0).sum())

    cm = confusion_matrix(y_test, y_pred).tolist()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    desc = {
        "total_records": len(X_test),
        "frauds_detected": fraud_count,
        "legitimate": legit_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    stats = [
        len(X_test), fraud_count, legit_count, cm, accuracy, precision, recall, f1, None
    ]

    frauds = []  # Could add fraud examples if needed

    return 1, desc, frauds, stats


def trainXGBSMOTE(train_data):
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    pipeline = Pipeline(steps=[
        ('smote', SMOTE(sampling_strategy=train_data["hyperparameters"]["smote_sampling_strategy"])),
        ('model', xgb.XGBClassifier(
        n_estimators=train_data["hyperparameters"]["n_estimators"],
        max_depth=train_data["hyperparameters"]["max_depth"],
        learning_rate=train_data["hyperparameters"]["learning_rate"],
        tree_method='hist',
        subsample=train_data["hyperparameters"]["subsample"],
        eval_metric=train_data["hyperparameters"]["eval_metric"]
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Save model
    model_path = f"app/models/{train_data['model_name']}.sav"
    joblib.dump(pipeline, model_path)

    # Update registry
    wr.updateModelRegistry(train_data['model_name'], model_path, "Custom XGBoost with SMOTE model")

    # Final prediction check
    y_pred = pipeline.predict(X_test)
    fraud_count = int((y_pred == 1).sum())
    legit_count = int((y_pred == 0).sum())

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred).tolist()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    desc = {
        "total_records": len(X_test),
        "frauds_detected": fraud_count,
        "legitimate": legit_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    stats = [
        len(X_test), fraud_count, legit_count, cm, accuracy, precision, recall, f1, None
    ]

    frauds = []

    return 1, desc, frauds, stats


def trainEnsemble(train_data):
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    smote = SMOTE(sampling_strategy=train_data["hyperparameters"]["smote_sampling_strategy"])

    X_train_n, y_train_n = smote.fit_resample(X_train, y_train)

    estimators = [
        ('rf', RandomForestClassifier(
        max_depth = train_data["hyperparameters"]["rf_max_depth"],
        n_estimators = train_data["hyperparameters"]["rf_n_estimators"])),
        ('xgb', xgb.XGBClassifier(
        n_estimators=train_data["hyperparameters"]["xgb_n_estimators"],
        max_depth=train_data["hyperparameters"]["xgb_max_depth"],
        learning_rate=train_data["hyperparameters"]["xgb_learning_rate"],
        tree_method='hist',
        ))
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1
    )

    stack.fit(X_train_n, y_train_n)

    # Save model
    model_path = f"app/models/{train_data['model_name']}.sav"
    joblib.dump(stack, model_path)

    # Update registry
    wr.updateModelRegistry(train_data['model_name'], model_path, "Custom Ensemble model")

    # Final prediction check
    y_pred = stack.predict(X_test)
    fraud_count = int((y_pred == 1).sum())
    legit_count = int((y_pred == 0).sum())

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred).tolist()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    desc = {
        "total_records": len(X_test),
        "frauds_detected": fraud_count,
        "legitimate": legit_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    stats = [
        len(X_test), fraud_count, legit_count, cm, accuracy, precision, recall, f1, None
    ]

    frauds = []

    return 1, desc, frauds, stats


