{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0ee3e571-80ce-47be-bea1-0e68a9f1b49b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-09-15 18:00:04.861 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\yigit\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st, pandas as pd, requests\n",
    "\n",
    "st.set_page_config(page_title=\"Explainable Risk Scoring\", layout=\"wide\")\n",
    "st.title(\"Explainable Multimodal Risk Scoring\")\n",
    "\n",
    "api = st.text_input(\"API URL\", \"http://localhost:8000/score\")\n",
    "file = st.file_uploader(\"Upload CSV with model features\", type=\"csv\")\n",
    "\n",
    "if file:\n",
    "    df = pd.read_csv(file)\n",
    "    st.write(\"Preview:\", df.head())\n",
    "\n",
    "    if st.button(\"Score all rows\"):\n",
    "        results = []\n",
    "        for _, r in df.iterrows():\n",
    "            payload = {\"data\": r.to_dict()}\n",
    "            res = requests.post(api, json=payload, timeout=30).json()\n",
    "            results.append(res)\n",
    "        out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)\n",
    "        st.success(\"Scoring complete\")\n",
    "        st.dataframe(out)\n",
    "\n",
    "        \n",
    "        row_ix = st.number_input(\"Explain row index\", min_value=0, max_value=len(out)-1, value=0, step=1)\n",
    "        if \"reasons\" in out.columns and isinstance(out.loc[row_ix, \"reasons\"], list):\n",
    "            st.subheader(\"Top reasons\")\n",
    "            reasons = out.loc[row_ix, \"reasons\"]\n",
    "            st.json(reasons)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d850bdc8-c4ae-4860-b092-0e94cdcb3713",
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
