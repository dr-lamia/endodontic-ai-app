import streamlit as st
import pandas as pd
import random
import requests
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import io
import os
import datetime
import openai

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Endodontic Multimodal AI Assistant", layout="wide")

# --- API Keys ---
HF_TOKEN = st.secrets.get("HF_TOKEN")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# --- 1. Generate + Train Model on Synthetic Dataset ---
@st.cache_data
def generate_and_process_dataset():
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

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    fs_model = RandomForestClassifier(n_estimators=100, random_state=42)
    fs_model.fit(X_resampled, y_resampled)
    selector = SelectFromModel(fs_model, threshold="median", prefit=True)
    X_selected = selector.transform(X_resampled)
    selected_feats = list(X.columns[selector.get_support()])

    df_final = pd.DataFrame(X_selected, columns=selected_feats)
    df_final["Diagnosis"] = y_resampled

    return df_final, selected_feats

df_data, selected_features = generate_and_process_dataset()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(df_data[selected_features], df_data["Diagnosis"])

# --- 2. App Interface ---
st.title("🦷 AI Dental Diagnosis Assistant")

st.sidebar.header("Patient Signs & Symptoms")
manual_feats = ["Radiolucency on X-ray"]
all_feats = sorted(set(selected_features + manual_feats))
inputs = {feat: st.sidebar.checkbox(feat) for feat in all_feats}

if st.sidebar.button("Predict Diagnosis"):
    input_df = pd.DataFrame([{k: inputs.get(k, 0) for k in selected_features}])
    diagnosis = rf_model.predict(input_df)[0]
    st.success(f"📋 Predicted Diagnosis: **{diagnosis}**")

    treatment_paths = {
        "Hyperemia": "Remove irritants, monitor, apply desensitizing agent.",
        "Acute Pulpitis": "Emergency pulpotomy or RCT, prescribe analgesics.",
        "Chronic Pulpitis": "Schedule root canal therapy and final restoration.",
        "Periapical Abscess": "Drain if needed, start antibiotics, perform RCT or extraction."
    }
    st.info(f"💊 Treatment Plan: {treatment_paths.get(diagnosis, 'No recommendation.')}")

    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    st.markdown("### 🔍 Feature Importance (Diagnosis Model)")
    st.dataframe(feat_imp_df, use_container_width=True)

    st.session_state["diagnosis"] = diagnosis

# --- 3. Chat Interface with OpenAI ---
st.markdown("---")
st.header("💬 Chat with AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the diagnosis or treatment:")
if user_input:
    diagnosis = st.session_state.get("diagnosis", "a dental condition")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful dental AI assistant."},
            {"role": "user", "content": f"Diagnosis: {diagnosis}. Question: {user_input}"}
        ]
    )
    answer = response.choices[0].message.content

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", answer))

# Display chat history
for sender, message in st.session_state.chat_history[-6:]:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 AI:** {message}")

if st.button("📝 Export Chat as .txt"):
    chat_lines = [f"{sender}: {message}" for sender, message in st.session_state.chat_history]
    chat_text = "\n".join(chat_lines)
    st.download_button("📥 Download Chat Log", chat_text, file_name="chat_history.txt")

if st.button("📁 Save Full Patient Case"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    diagnosis = st.session_state.get("diagnosis", "unknown")
    symptoms = [feat for feat, val in st.session_state.items() if feat in selected_features and val]
    chat_lines = [f"{sender}: {message}" for sender, message in st.session_state.chat_history]

    case_text = f"Patient Case - {timestamp}\nDiagnosis: {diagnosis}\nSymptoms: {', '.join(symptoms)}\n\nChat:\n" + "\n".join(chat_lines)
    st.download_button("📥 Download Full Case Log", case_text, file_name=f"patient_case_{timestamp}.txt")
