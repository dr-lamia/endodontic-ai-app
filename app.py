
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import torch
from PIL import Image
import random

@st.cache_data
def generate_and_process_dataset():
    signs_symptoms = {
        "Spontaneous Pain": [0, 1],
        "Pain on Biting": [0, 1],
        "Sensitivity to Cold": [0, 1],
        "Sensitivity to Heat": [0, 1],
        "Swelling": [0, 1],
        "Sinus Tract": [0, 1],
        "Radiolucency on X-ray": [0, 1],
        "Tooth Discoloration": [0, 1],
        "Percussion Sensitivity": [0, 1],
        "Palpation Sensitivity": [0, 1],
        "Deep Caries": [0, 1],
        "Previous Restoration": [0, 1],
        "Mobility": [0, 1],
        "No Response to Vitality Test": [0, 1],
        "Lingering Pain": [0, 1]
    }
    conditions = ["Hyperemia", "Acute Pulpitis", "Chronic Pulpitis", "Periapical Abscess"]
    data = []
    for _ in range(500):
        patient = {symptom: random.choice(values) for symptom, values in signs_symptoms.items()}
        if patient["Sensitivity to Cold"] and not patient["Lingering Pain"]:
            condition = "Hyperemia"
        elif patient["Lingering Pain"] and patient["Pain on Biting"]:
            condition = "Acute Pulpitis"
        elif patient["Lingering Pain"] and not patient["Pain on Biting"]:
            condition = "Chronic Pulpitis"
        elif patient["Swelling"] or patient["Sinus Tract"] or patient["Radiolucency on X-ray"]:
            condition = "Periapical Abscess"
        else:
            condition = random.choice(conditions)
        patient["Diagnosis"] = condition
        data.append(patient)
    df = pd.DataFrame(data)
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    model_fs = RandomForestClassifier(n_estimators=100, random_state=42)
    model_fs.fit(X_resampled, y_resampled)
    selector = SelectFromModel(model_fs, threshold="median", prefit=True)
    X_selected = selector.transform(X_resampled)
    selected_features = list(X.columns[selector.get_support()])
    df_final = pd.DataFrame(X_selected, columns=selected_features)
    df_final["Diagnosis"] = y_resampled

    return df_final, selected_features

@st.cache_resource

df_data, selected_features = generate_and_process_dataset()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df_data[selected_features], df_data["Diagnosis"])

st.title("🦷 Endodontic Disease Diagnosis AI App")
st.write("Select the signs and symptoms from the list below and click **Predict Diagnosis** to get the AI's recommendation.")

inputs = {}
st.sidebar.header("Patient Signs & Symptoms")
for symptom in selected_features:
    inputs[symptom] = st.sidebar.checkbox(symptom)

if st.sidebar.button("Predict Diagnosis"):
    input_df = pd.DataFrame([inputs])
    diagnosis = model.predict(input_df)[0]
    st.success(f"**Predicted Diagnosis:** {diagnosis}")

    treatment_paths = {
        "Hyperemia": "Monitor and remove irritants. Use desensitizing agents. Follow-up recommended.",
        "Acute Pulpitis": "Perform emergency pulpotomy or pulpectomy. Prescribe analgesics. Plan for root canal therapy.",
        "Chronic Pulpitis": "Schedule root canal therapy. Consider crown restoration after treatment.",
        "Periapical Abscess": "Drain abscess if necessary. Start antibiotics. Perform root canal therapy or extraction."
    }
    treatment = treatment_paths.get(diagnosis, "No treatment path available.")
    st.info(f"**Suggested Treatment Path:** {treatment}")

st.markdown("---")
st.write("**Feature importance (for transparency):**")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=selected_features).sort_values(ascending=False)
st.write(feat_imp)

st.markdown("---")
st.subheader("📷 Optional: Upload Dental X-ray for MedGemma AI Analysis")
xray_file = st.file_uploader("Upload a dental X-ray image (JPG or PNG)", type=["jpg", "jpeg", "png"])

def query_medgemma_api(image_bytes):
    import requests
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    API_URL = "https://api-inference.huggingface.co/models/google/medgemma-2b"
    response = requests.post(API_URL, headers=headers, files={"inputs": image_bytes})
    return response.json()
if xray_file is not None:
    st.image(xray_file, caption="Uploaded X-ray", use_column_width=True)
    response = query_medgemma_api(xray_file.read())
    st.success("**MedGemma AI Interpretation:**")
    if isinstance(response, dict):
        st.json(response)
    else:
        st.write(response)
