
import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from PIL import Image

def query_huggingface_model(image_bytes, model_id, retries=2):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, files={"inputs": image_bytes})
            if response.ok:
                return response.json()
        except Exception as e:
            continue
    return {"error": f"{model_id} failed after {retries} attempts."}

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
    X_res, y_res = smote.fit_resample(X, y)
    fs_model = RandomForestClassifier(random_state=42)
    fs_model.fit(X_res, y_res)
    selector = SelectFromModel(fs_model, threshold="median", prefit=True)
    X_sel = selector.transform(X_res)
    sel_features = list(X.columns[selector.get_support()])
    df_final = pd.DataFrame(X_sel, columns=sel_features)
    df_final["Diagnosis"] = y_res
    return df_final, sel_features

df_data, selected_features = generate_and_process_dataset()
model = RandomForestClassifier()
model.fit(df_data[selected_features], df_data["Diagnosis"])

st.title("🦷 Endodontic AI App")

inputs = {}
st.sidebar.header("Patient Signs & Symptoms")
manual_features = ["Radiolucency on X-ray"]
for feat in sorted(set(selected_features + manual_features)):
    inputs[feat] = st.sidebar.checkbox(feat)

if st.sidebar.button("Predict Diagnosis"):
    input_df = pd.DataFrame([{k: inputs.get(k, 0) for k in selected_features}])
    diagnosis = model.predict(input_df)[0]
    st.success(f"📋 Predicted Diagnosis: **{diagnosis}**")

    # --- Feature Importance ---
    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)
    st.markdown("### 🔍 Feature Importance (Diagnosis Model)")
    st.dataframe(feat_df.reset_index(drop=True), use_container_width=True)

import os
import requests
import base64
import google.generativeai as genai

HF_TOKEN = os.getenv("HF_TOKEN", st.secrets.get("HF_TOKEN", ""))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY", ""))

# Gemini setup
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

st.subheader("📷 Optional: Upload Dental X-ray for AI Interpretation")
vision_model = st.selectbox("Choose Vision-Language Model", ["Gemini Vision", "OpenAI GPT-4 Vision", "MedGemma", "BLIP (captioning)", "ViT-GPT2 (captioning)"])
xray_file = st.file_uploader("Upload Dental X-ray image (JPG or PNG)", type=["jpg", "jpeg", "png"])

def query_huggingface_model(image_bytes, model_id, retries=2):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, files={"inputs": image_bytes})
            if response.ok:
                return response.json()
        except Exception:
            continue
    return {"error": f"{model_id} failed after {retries} attempts."}

def analyze_xray(image):
    caption = ""
    img = Image.open(image).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    try:
        if vision_model == "Gemini Vision" and GOOGLE_API_KEY:
            model = genai.GenerativeModel("gemini-pro-vision")
            resp = model.generate_content([img, "Describe this dental X-ray"])
            caption = resp.text
        elif vision_model == "OpenAI GPT-4 Vision":
            caption = "GPT-4 Vision not implemented in this offline version."
        elif vision_model == "MedGemma":
            result = query_huggingface_model(image.read(), "google/medgemma-2b")
            caption = result[0].get("generated_text", result.get("error", "MedGemma failed."))
        elif vision_model == "BLIP (captioning)":
            result = query_huggingface_model(image.read(), "Salesforce/blip-image-captioning-base")
            caption = result[0].get("generated_text", result.get("error", "BLIP failed."))
        elif vision_model == "ViT-GPT2 (captioning)":
            result = query_huggingface_model(image.read(), "nlpconnect/vit-gpt2-image-captioning")
            caption = result[0].get("generated_text", result.get("error", "ViT-GPT2 failed."))
    except Exception as e:
        caption = f"Exception: {str(e)}"

    return caption

if xray_file and st.button("Analyze X-ray"):
    st.subheader("🧠 AI X-ray Interpretation:")
    output = analyze_xray(xray_file)
    st.success(output) if "failed" not in output.lower() else st.error(output)

    if "Diagnosis" in locals():
        if diagnosis.lower() in output.lower():
            st.success(f"✅ The AI output supports the diagnosis of **{diagnosis}**")
        else:
            st.warning(f"⚠️ The X-ray does **not clearly support** the diagnosis of **{diagnosis}**.")
