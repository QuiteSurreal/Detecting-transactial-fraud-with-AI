import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

from app.services import predict as pred
from app.services import train as train

EXPECTED_SCHEMA = {
    "step": int,
    "type": str,
    "amount": float,
    "nameOrig": str,
    "oldbalanceOrg": float,
    "newbalanceOrig": float,
    "nameDest": str,
    "oldbalanceDest": float,
    "newbalanceDest": float,
    "isFraud": int,
}

def preprocessFile(data, model_name):
    try:
        dfRaw = pd.read_csv(data, delimiter = ',', nrows = 100000)
    except Exception as e:
        return 0, [f"Failed to read CSV file: {str(e)}"], [], None
    
    df = dfRaw.copy()

    y_true = None
    if ('isFraud' in df):
        y_true = df['isFraud'].values
        df = df.drop('isFraud', axis=1)

    errors = validateData(df)

    if (errors):
        return 0, errors, [], None

    try:
        df = preprocess(df, model_name)
    except Exception as e:
        return 0, [f"Error preprocessing data: {str(e)}"], [], None
    

    try:
        y_pred = pred.runPrediction(model_name, df)
    except Exception as e:
        return 0, [f"Error running prediction: {str(e)}"], [], None

    if (model_name != "Isolation Forest + KMeans"):
        dfRaw['prediction'] = y_pred
        fraud_count = int((dfRaw["prediction"] == 1).sum())
        legit_count = int((dfRaw["prediction"] == 0).sum())

        frauds = dfRaw[dfRaw["prediction"] == 1]

        desc = {
            "total_records": len(dfRaw),
            "frauds_detected": fraud_count,
            "legitimate": legit_count
        }

        stats = None
        if (y_true is not None):
            cm = confusion_matrix(y_true, y_pred).tolist()
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            stats = [
                len(dfRaw), fraud_count, legit_count, cm, accuracy, precision, recall, f1
            ]

        return 1, desc, frauds.to_dict(orient='records'), stats
    else:
        dfRaw['is_anomaly'] = [pred['is_anomaly'] for pred in y_pred]
        dfRaw['cluster'] = [pred['cluster'] for pred in y_pred]

        anomalous_entries = dfRaw[dfRaw['is_anomaly'] == True].copy()
        anomaly_count = len(anomalous_entries)
        normal_count = len(dfRaw) - anomaly_count
        cluster_count = 3
        desc = {
            "total_records": len(dfRaw),
            "anomalies_detected": anomaly_count,
            "normal": normal_count,
            "clusters": cluster_count
        }

        return 1, desc, anomalous_entries.to_dict(orient='records'), None
    

def preprocess(df: pd.DataFrame, model_name):

    if ('isFlaggedFraud' in df):
        df = df.drop('isFlaggedFraud', axis=1)

    df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

    for col in df.select_dtypes(include=['object']).columns:
        col = col.strip()
        df[col] = LabelEncoder().fit_transform(df[col])

    df = df.astype(float)

    if (model_name == "Isolation Forest + KMeans"):
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


    return df

def preprocessForTrain(data):
    try:
        dfRaw = pd.read_csv(data, delimiter=',', nrows=100000)
    except Exception as e:
        return None, [f"Failed to read CSV file: {str(e)}"]
    
    if 'isFraud' not in dfRaw.columns:
        return None, ["Missing 'isFraud' column in training data"]
    
    y = dfRaw['isFraud'].values
    df = dfRaw.drop('isFraud', axis=1)
    
    errors = validateData(df)
    if errors:
        return None, errors
    
    df = preprocess(df, "xgb")
    return df, y


def validateData(data: pd.DataFrame):
    errors = []

    for col, expectedType in EXPECTED_SCHEMA.items():
        if col not in data.columns:
            if col == 'isFraud':
                continue
            errors.append(f"Missing columns: {col}")
            continue

        if not data[col].map(lambda x: isinstance(x, expectedType)).all():
            errors.append(f"Column {col} has invalid type (expected type: {expectedType.__name__})")
    
    return errors
