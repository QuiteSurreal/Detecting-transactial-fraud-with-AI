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
from app.services import preprocess as prep
from app.services import predict as pred
from io import StringIO






def prepareTrain(train_data, selected_model, mode, file_data=None):
    """
    Loads data, preprocesses, splits, and delegates to appropriate trainer.
    
    Args:
        train_data (dict): Configuration with model_name and hyperparameters
        selected_model (str): Base model type ('xgb', 'xgb_smote', 'ensemble')
        mode (int): 0=new training, 1=model upgrade
        file_data (bytes): Optional CSV file data for upload-based training
    
    Returns:
        tuple: (success, desc, frauds, stats) - success flag, task description, fraud records, evaluation stats
    """
    if file_data and mode == 1:
        file_data = StringIO(file_data.decode("utf-8"))
        try:
            dfRaw = pd.read_csv(file_data, delimiter=',', nrows=100000)
        except Exception as e:
            return 0, [f"Failed to read CSV file: {str(e)}"], [], None
    else:
        try:
            dfRaw = pd.read_csv('./resources/data/Input.csv', delimiter=',', nrows=100000)
        except Exception as e:
            return 0, [f"Failed to read CSV file: {str(e)}"], [], None

    if 'isFraud' not in dfRaw.columns:
        return 0, ["Missing 'isFraud' column in training data"], [], None

    y = dfRaw['isFraud'].values
    dfRaw = dfRaw.drop('isFraud', axis=1)

    errors = prep.validateData(dfRaw)
    if errors:
        return 0, errors, [], None

    df = prep.preprocess(dfRaw.copy(), "xgb")
    if df is None:
        return 0, ["Failed to preprocess training data"], [], None
    
    raw_train, raw_test = train_test_split(
        dfRaw,
        test_size=0.2
    )

    X_train, X_test, y_train, y_test= train_test_split(
        df,
        y,
        stratify=y,
        test_size=0.2
    )

    if (mode == 1):
        return upgradeXGB(train_data, X_train, X_test, y_train, y_test, raw_test)
    elif (selected_model == "xgb"):
        return trainXGB(train_data, X_train, X_test, y_train, y_test, raw_test)
    elif (selected_model == "xgb_smote"):
        return trainXGBSMOTE(train_data, X_train, X_test, y_train, y_test, raw_test)
    elif (selected_model == "ensemble"):
        return trainEnsemble(train_data, X_train, X_test, y_train, y_test, raw_test)
    else:
        return 0, [], [], []

def trainXGB(train_data, X_train, X_test, y_train, y_test, raw_test):
    """
    Trains a standalone XGBoost classifier without oversampling.
    
    Args:
        train_data (dict): Contains model_name and hyperparameters
        X_train, X_test: Preprocessed feature data for training and testing
        y_train, y_test: Labels for training and testing
        raw_test: Original unpreprocessed test records for fraud display
    
    Returns:
        tuple: (success=1, desc, frauds, stats) - results with evaluation metrics
    """
    model = xgb.XGBClassifier(
        n_estimators=train_data["hyperparameters"]["n_estimators"],
        max_depth=train_data["hyperparameters"]["max_depth"],
        learning_rate=train_data["hyperparameters"]["learning_rate"],
        scale_pos_weight=train_data["hyperparameters"]["scale_pos_weight"],
        tree_method='hist',
        eval_metric=train_data["hyperparameters"]["eval_metric"]
    )

    model.fit(X_train, y_train)

    model_path = f"app/models/{train_data['model_name']}.ubj"
    model.save_model(model_path)

    wr.updateModelRegistry(
        train_data['model_name'], 
        model_path, 
        "Custom XGB model", 
        "xgb", 
        1,
        train_data["hyperparameters"]
    )

    y_pred = model.predict(X_test)

    exp = pred.makeExplanation(model, X_test)
    
    desc, frauds, stats = evalModel(y_pred, X_test, y_test, exp, raw_test)

    return 1, desc, frauds, stats


def trainXGBSMOTE(train_data, X_train, X_test, y_train, y_test, raw_test):
    """
    Trains XGBoost with SMOTE oversampling for imbalanced data handling.
    
    Args:
        train_data (dict): Contains model_name and hyperparameters
        X_train, X_test: Preprocessed feature data for training and testing
        y_train, y_test: Labels for training and testing
        raw_test: Original unpreprocessed test records for fraud display
    
    Returns:
        tuple: (success=1, desc, frauds, stats) - results with evaluation metrics
    """
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

    model_path = f"app/models/{train_data['model_name']}.ubj"
    pipeline.named_steps['model'].save_model(model_path)

    wr.updateModelRegistry(
        train_data['model_name'], 
        model_path, 
        "Custom XGBoost with SMOTE model", 
        "xgb_smote", 
        1,
        train_data["hyperparameters"]
    )

    y_pred = pipeline.predict(X_test)

    exp = pred.makeExplanation(pipeline.named_steps['model'], X_test)
    
    desc, frauds, stats = evalModel(y_pred, X_test, y_test, exp, raw_test)

    return 1, desc, frauds, stats

def upgradeXGB(upgrade_data, X_train, X_test, y_train, y_test, raw_test):
    """
    Fine-tunes an existing XGBoost model with new training data.
    
    Args:
        upgrade_data (dict): Contains model_path, model_name, base_model, and hyperparameters
        X_train, X_test: Preprocessed feature data for training and testing
        y_train, y_test: Labels for training and testing
        raw_test: Original unpreprocessed test records for fraud display
    
    Returns:
        tuple: (success, desc, frauds, stats) or (0, [error], [], None) on failure
    """


    try:
        model = xgb.XGBClassifier()
        model.load_model(upgrade_data['model_path'])
    except Exception as e:
        return 0, [f"Failed to load model: {str(e)}"], [], None

    if "hyperparameters" in upgrade_data and upgrade_data["hyperparameters"]:
        try:
            model.set_params(**upgrade_data["hyperparameters"])
        except Exception as e:
            print(f"Warning: Could not update some params: {e}")

    if (upgrade_data["base_model"] == "xgb_smote"):
        smote = SMOTE(sampling_strategy=upgrade_data["hyperparameters"])
        X_train, y_train = smote.fit_resample(X_train, y_train)

    try:
        model.fit(X_train, y_train, xgb_model=model.get_booster())
    except Exception as e:
        return 0, [f"Error training model: {str(e)}"], [], None

    model_path = f"app/models/{upgrade_data['model_name']}.ubj"
    model.save_model(model_path)

    wr.updateModelRegistry(
        upgrade_data['model_name'], 
        model_path, 
        "Upgraded XGB model", 
        upgrade_data["base_model"], 
        1,
        upgrade_data.get("hyperparameters", {})
    )

    y_pred = model.predict(X_test)

    exp = pred.makeExplanation(model, X_test)
    
    desc, frauds, stats = evalModel(y_pred, X_test, y_test, exp, raw_test)

    return 1, desc, frauds, stats



def trainEnsemble(train_data, X_train, X_test, y_train, y_test, raw_test):
    """
    Trains a stacking ensemble combining Random Forest and XGBoost with Logistic Regression meta-learner.
    
    Args:
        train_data (dict): Contains model_name and hyperparameters
        X_train, X_test: Preprocessed feature data for training and testing
        y_train, y_test: Labels for training and testing
        raw_test: Original unpreprocessed test records for fraud display
    
    Returns:
        tuple: (success=1, desc, frauds, stats) - results with evaluation metrics
    """
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

    model_path = f"app/models/{train_data['model_name']}.sav"
    joblib.dump(stack, model_path)

    wr.updateModelRegistry(
        train_data['model_name'], 
        model_path, 
        "Custom Ensemble model", 
        "ensemble", 
        0,
        train_data["hyperparameters"]
    )

    y_pred = stack.predict(X_test)

    exp = pred.makeExplanation(stack.estimators_[1], X_test)
    
    desc, frauds, stats = evalModel(y_pred, X_test, y_test, exp, raw_test)
    

    return 1, desc, frauds, stats

def evalModel(y_pred, X_test, y_test, exp, raw_test=None):
    """
    Evaluates model predictions and compiles performance metrics and fraud records.
    
    Args:
        y_pred: Model predictions (0/1 for fraud/legit)
        X_test: Preprocessed test features
        y_test: True test labels
        exp (dict): Feature importance explanations from SHAP
        raw_test (DataFrame, optional): Original unpreprocessed test records for fraud display
    
    Returns:
        tuple: (desc, frauds, stats) - description dict, fraud records list, metrics stats list
    """
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
        "feature_importance": exp
    }

    stats = [
        len(X_test), fraud_count, legit_count, cm, accuracy, precision, recall, f1, exp
    ]

    if raw_test is not None:
        X_eval = raw_test.copy()
    else:
        X_eval = X_test.copy()

    X_eval['prediction'] = y_pred
    frauds = X_eval[X_eval['prediction'] == 1]

    return desc, frauds.to_dict(orient='records'), stats