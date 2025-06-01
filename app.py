import streamlit as st
import pandas as pd
import random
import requests
import openai
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from PIL import Image

# --- API Keys ---
HF_TOKEN = st.secrets.get("HF_TOKEN")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# --- Diagnosis Data Preparation ---
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

df_data, selected_features = generate_and_process_dataset()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df_data[selected_features], df_data["Diagnosis"])

# --- UI Section ---
st.title("🦷 Endodontic AI Assistant with Multimodal X-ray Analysis")

st.header("🧾 Diagnosis & Treatment Recommendation")

inputs = {}
st.sidebar.header("Patient Signs & Symptoms")
manual_features = ["Radiolucency on X-ray"]
all_features = list(set(selected_features + manual_features))
for symptom in sorted(all_features):
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

st.write("**🔍 Feature Importance:**")
feat_imp = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
st.write(feat_imp)

# --- Multimodal AI Section ---
st.markdown("---")
st.header("🧠 X-ray Interpretation via Vision-Language Models")

model_choice = st.selectbox("Choose Vision-Language Model", [
    "MedGemma (clinical)", 
    "BLIP (captioning)", 
    "ViT-GPT2 (captioning)", 
    "OpenAI GPT-4 Vision", 
    "Gemini Vision"
])

xray_file = st.file_uploader("Upload Dental X-ray", type=["jpg", "jpeg", "png"])

def medgemma_api(image_bytes):
    url = "https://api-inference.huggingface.co/models/google/medgemma-2b"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(url, headers=headers, files={"inputs": image_bytes})
    try:
        return response.json()
    except Exception:
        return {"error": "MedGemma API failed or returned invalid response."}

def blip_api(image_bytes):
    url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(url, headers=headers, files={"inputs": image_bytes})
    try:
        return response.json()
    except Exception:
        return {"error": "BLIP API failed or returned invalid response."}

def vit_gpt2_api(image_bytes):
    url = "https://api-inference.huggingface.co/models/nielsr/vit-gpt2-image-captioning"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(url, headers=headers, files={"inputs": image_bytes})
    try:
        return response.json()
    except Exception:
        return {"error": "ViT-GPT2 API failed or returned invalid response."}

def openai_vision_api(image_bytes):
    return {"error": "Not implemented in this offline version."}

def gemini_api(image):
    return {"error": "Not implemented in this offline version."}

if xray_file:
    image = Image.open(xray_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    image_bytes = xray_file.read()

    if st.button("Analyze X-ray"):
        with st.spinner(f"Analyzing with {model_choice}..."):
            if model_choice.startswith("MedGemma"):
                response = medgemma_api(image_bytes)
            elif model_choice.startswith("BLIP"):
                response = blip_api(image_bytes)
            elif model_choice.startswith("ViT"):
                response = vit_gpt2_api(image_bytes)
            elif model_choice.startswith("OpenAI"):
                response = openai_vision_api(image_bytes)
            elif model_choice.startswith("Gemini"):
                response = gemini_api(image)
            else:
                response = {"error": "Unknown model."}

        if isinstance(response, dict) and "error" in response:
            st.error(response["error"])
        else:
            st.success("✅ Multimodal AI Response:")
            st.write(response)
