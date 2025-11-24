import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


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

def preprocessFile(data, modelName, mode):
    try:
        dfRaw = pd.read_csv(data, delimiter = ',', nrows = 100000)
    except Exception as e:
        return 0, [f"Failed to read CSV file: {str(e)}"], []
    
    df = dfRaw.copy()

    if ('isFlaggedFraud' in df):
        df = df.drop('isFlaggedFraud', axis=1)

    errors = validateData(df)

    if (errors):
        return 0, errors, []

    try:
        df = preprocess(df)
    except Exception as e:
        return 0, [f"Error preprocessing data: {str(e)}"], []
    
    print("prep done")

    try:
        result = pred.runPrediction(modelName, df)
    except Exception as e:
        return 0, [f"Error running prediction: {str(e)}"], []

    dfRaw['prediction'] = result
    fraud_count = int((dfRaw["prediction"] == 1).sum())
    legit_count = int((dfRaw["prediction"] == 0).sum())

    frauds = dfRaw[dfRaw["prediction"] == 1]

    desc = {
        "total_records": len(dfRaw),
        "frauds_detected": fraud_count,
        "legitimate": legit_count
    }

    # Save only frauds for inspection


    print("result got")

    return 1, desc, frauds.to_dict(orient='records')

def preprocessJSON(request, mode):
    import time
    time.sleep(10)
    return 1, "fraudulent: 10, rows: 1000"

def preprocess(df: pd.DataFrame):
    
    df = df.drop('isFraud', axis=1)
    df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

    for col in df.select_dtypes(include=['object']).columns:
        col = col.strip()
        df[col] = LabelEncoder().fit_transform(df[col])

    df = df.astype(float)

    return df


def validateData(data: pd.DataFrame):
    errors = []

    print("dod youd ided lid bod")

    for col in EXPECTED_SCHEMA:
        if col not in data.columns:
            errors.append(f"Missing columns: {col}")
    
    for col, expectedType in EXPECTED_SCHEMA.items():
        if col in data.columns:
            if not data[col].map(lambda x: isinstance(x, expectedType)).all():
                errors.append(f"Column {col} has invalid type (expected type: {expectedType.__name__})")
    
    return errors