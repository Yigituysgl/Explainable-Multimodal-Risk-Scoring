
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List

import joblib, json, pandas as pd


MODEL_PATH = "xgb_model.joblib"
SCALER_PATH = "scaler.joblib"          # if you used StandardScaler; ok if missing
FEATS_PATH = "feature_list.json"
THR_PATH   = "threshold.json"


model = joblib.load(MODEL_PATH)

try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

feature_order: List[str] = json.load(open(FEATS_PATH))
THRESHOLD = float(json.load(open(THR_PATH))["threshold"])


app = FastAPI(title="Risk Scoring API")

class Payload(BaseModel):
    data: Dict[str, Any]

try:
    import shap
    explainer = shap.TreeExplainer(model)
except Exception:
    explainer = None

def _to_frame(d: Dict[str, Any]) -> pd.DataFrame:
    """Make a 1-row dataframe with correct column order; fill missing with 0."""
    df = pd.DataFrame([d])
    df = df.reindex(columns=feature_order, fill_value=0)
    if scaler is not None:
        df[feature_order] = scaler.transform(df[feature_order])
    return df

def _explain_row(df_row: pd.Series, top_k: int = 5):
    if explainer is None:
        return []
    sv = explainer.shap_values(df_row.values.reshape(1, -1))[0]
    pairs = sorted(zip(feature_order, sv), key=lambda t: abs(t[1]), reverse=True)[:top_k]
    return [{"feature": f, "impact": float(v)} for f, v in pairs]

@app.get("/health")
def health():
    return {"status": "ok", "features": feature_order[:5]}

@app.post("/score")
def score(p: Payload):
    df = _to_frame(p.data)
    proba = float(model.predict_proba(df)[:, 1][0])
    label = int(proba >= THRESHOLD)
    reasons = _explain_row(df.iloc[0])

    return {
        "default_proba": proba,
        "class": label,
        "threshold": THRESHOLD,
        "reasons": reasons,
    }
