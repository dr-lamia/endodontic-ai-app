import streamlit as st
import pandas as pd
import random
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from PIL import Image

# --- API Keys (from Streamlit secrets) ---
HF_TOKEN = st.secrets.get("HF_TOKEN")
# (BLIP is public once HF_TOKEN is valid; no other keys needed here)

# --- 1. Generate + Train on Synthetic Dataset ---
@st.cache_data
def generate_and_process_dataset():
    # Define all possible signs & symptoms
    symptoms = {
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
        patient = {symptom: random.choice(vals) for symptom, vals in symptoms.items()}

        # Simple rules to assign a “true” diagnosis
        if patient["Sensitivity to Cold"] and not patient["Lingering Pain"]:
            condition = "Hyperemia"
        elif patient["Lingering Pain"] and patient["Pain on Biting"]:
            condition = "Acute Pulpitis"
        elif patient["Lingering Pain"]:
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

    # Balance classes via SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Feature selection via RandomForest
    fs_model = RandomForestClassifier(n_estimators=100, random_state=42)
    fs_model.fit(X_resampled, y_resampled)
    selector = SelectFromModel(fs_model, threshold="median", prefit=True)
    X_selected = selector.transform(X_resampled)
    selected_feats = list(X.columns[selector.get_support()])

    df_final = pd.DataFrame(X_selected, columns=selected_feats)
    df_final["Diagnosis"] = y_resampled

    return df_final, selected_feats

# Generate dataset and train final classifier
df_data, selected_features = generate_and_process_dataset()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(df_data[selected_features], df_data["Diagnosis"])

# --- 2. UI: Symptom Input & Diagnosis Prediction ---
st.title("🦷 Endodontic AI Assistant with Free VLM")

st.sidebar.header("Patient Signs & Symptoms")

# Always include “Radiolucency on X-ray” even if not selected by RF
manual_feats = ["Radiolucency on X-ray"]
all_feats = sorted(set(selected_features + manual_feats))

inputs = {}
for feat in all_feats:
    inputs[feat] = st.sidebar.checkbox(feat)

if st.sidebar.button("Predict Diagnosis"):
    # Build DataFrame with only the features the model expects
    input_df = pd.DataFrame([{k: inputs.get(k, 0) for k in selected_features}])
    diagnosis = rf_model.predict(input_df)[0]
    st.success(f"📋 Predicted Diagnosis: **{diagnosis}**")

    # Show treatment suggestion
    treatment_paths = {
        "Hyperemia": "Remove irritants, monitor, apply desensitizing agent.",
        "Acute Pulpitis": "Emergency pulpotomy or RCT, prescribe analgesics.",
        "Chronic Pulpitis": "Schedule root canal therapy and final restoration.",
        "Periapical Abscess": "Drain if needed, start antibiotics, perform RCT or extraction."
    }
    st.info(f"💊 Treatment Plan: {treatment_paths.get(diagnosis, 'No recommendation.')}")

    # Feature Importance Table
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    st.markdown("### 🔍 Feature Importance (Diagnosis Model)")
    st.dataframe(feat_imp_df, use_container_width=True)

    # Store diagnosis for later correlation
    st.session_state["diagnosis"] = diagnosis

