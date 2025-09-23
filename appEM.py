{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c0c5f326-b620-417f-9d39-bd2aff88836f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "from pydantic import BaseModel\n",
    "import joblib, json, pandas as pd\n",
    "\n",
    "app = FastAPI(title=\"Risk Scoring API\")\n",
    "\n",
    "MODEL_PATH = \"artifacts/xgb_model.joblib\"\n",
    "SCALER_PATH = \"artifacts/scaler.joblib\"\n",
    "FEATS_PATH  = \"artifacts/feature_list.json\"\n",
    "THR_PATH    = \"artifacts/threshold.json\"\n",
    "\n",
    "model = joblib.load(MODEL_PATH)\n",
    "scaler = joblib.load(SCALER_PATH)\n",
    "feature_order = json.load(open(FEATS_PATH))\n",
    "THRESHOLD = json.load(open(THR_PATH))[\"threshold\"]\n",
    "\n",
    "\n",
    "try:\n",
    "    import shap\n",
    "    explainer = shap.TreeExplainer(model)\n",
    "except Exception:\n",
    "    explainer = None\n",
    "\n",
    "class Payload(BaseModel):\n",
    "    data: dict  \n",
    "\n",
    "@app.get(\"/health\")\n",
    "def health():\n",
    "    return {\"status\": \"ok\", \"features\": feature_order[:5]}\n",
    "\n",
    "@app.post(\"/score\")\n",
    "def score(p: Payload):\n",
    "    X = pd.DataFrame([p.data])[feature_order]\n",
    "    Xs = pd.DataFrame(scaler.transform(X), columns=feature_order)\n",
    "    proba = float(model.predict_proba(Xs)[:,1][0])\n",
    "    label = int(proba >= THRESHOLD)\n",
    "\n",
    "    reasons = None\n",
    "    if explainer is not None:\n",
    "        sv = explainer.shap_values(Xs.values)[0]\n",
    "        pairs = sorted(zip(feature_order, sv), key=lambda t: abs(t[1]), reverse=True)[:5]\n",
    "        reasons = [{\"feature\": f, \"impact\": float(v)} for f, v in pairs]\n",
    "\n",
    "    return {\"risk_score\": proba, \"label\": label, \"threshold\": THRESHOLD, \"reasons\": reasons}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5cd2f7df-610c-423d-a8e0-869c69fde357",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
