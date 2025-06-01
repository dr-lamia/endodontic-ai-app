
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

# --- Dataset Prep ---
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

# --- UI ---
st.title("🦷 Endodontic AI App with Multimodal VLM Analysis")

inputs = {}
st.sidebar.header("Patient Signs & Symptoms")
manual_features = ["Radiolucency on X-ray"]
for feat in sorted(set(selected_features + manual_features)):
    inputs[feat] = st.sidebar.checkbox(feat)

if st.sidebar.button("Predict Diagnosis"):
    input_df = pd.DataFrame([{k: inputs.get(k, 0) for k in selected_features}])
    diagnosis = model.predict(input_df)[0]
    st.success(f"📋 Predicted Diagnosis: **{diagnosis}**")

    treatment = {
        "Hyperemia": "Remove irritants, monitor, apply desensitizing agent.",
        "Acute Pulpitis": "Emergency pulpotomy or RCT, analgesics.",
        "Chronic Pulpitis": "Scheduled root canal and restoration.",
        "Periapical Abscess": "Drainage, antibiotics, RCT or extraction."
    }.get(diagnosis, "No recommendation.")
    st.info(f"💊 Treatment Plan: {treatment}")

    st.session_state["diagnosis"] = diagnosis

# --- X-ray Upload and Analysis ---
st.markdown("---")
st.header("🧠 Analyze Dental X-ray with AI")

xray_model = st.selectbox("Choose Vision Model", [
    "MedGemma", "BLIP", "ViT-GPT2", "Gemini", "OpenAI GPT-4"
])

xray_file = st.file_uploader("Upload a Dental X-ray (JPG, PNG)", type=["jpg", "jpeg", "png"])
if xray_file:
    image = Image.open(xray_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    image_bytes = xray_file.read()

    def call_hf(model_id):
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            files={"inputs": image_bytes}
        )
        try:
            return response.json()
        except:
            return {"error": "API failed."}

    def call_gemini(image):
        model = genai.GenerativeModel("gemini-pro-vision")
        try:
            response = model.generate_content(["Analyze this dental X-ray for periapical or pulp disease.", image])
            return response.text
        except:
            return "Gemini failed."

    def call_openai(image):
        try:
            base64_img = openai._to_base64(image_bytes)
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this dental X-ray for periapical, pulpal or bony pathologies."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                max_tokens=512
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return "GPT-4 Vision failed."

    if st.button("Analyze X-ray"):
        with st.spinner(f"Analyzing using {xray_model}..."):
            if xray_model == "MedGemma":
                caption = call_hf("google/medgemma-2b")
            elif xray_model == "BLIP":
                caption = call_hf("Salesforce/blip-image-captioning-base")
            elif xray_model == "ViT-GPT2":
                caption = call_hf("nlpconnect/vit-gpt2-image-captioning")
            elif xray_model == "Gemini":
                caption = call_gemini(image)
            elif xray_model == "OpenAI GPT-4":
                caption = call_openai(image)
            else:
                caption = {"error": "Model not supported."}

        if isinstance(caption, dict) and "error" in caption:
            st.error(caption["error"])
        else:
            caption_text = caption if isinstance(caption, str) else caption[0].get("generated_text", "No output.")
            st.success("🧠 AI X-ray Interpretation:")
            st.write(caption_text)

            # Simple rule-based correlation logic
            diagnosis = st.session_state.get("diagnosis", "")
            correlation_keywords = {
                "Periapical Abscess": ["radiolucency", "lesion", "bone loss", "apex"],
                "Acute Pulpitis": ["pulp chamber", "widening", "dark area"],
                "Chronic Pulpitis": ["calcification", "chronic", "discoloration"],
                "Hyperemia": ["no visible lesion", "intact", "normal"]
            }
            matched_keywords = correlation_keywords.get(diagnosis, [])
            support = any(kw in caption_text.lower() for kw in matched_keywords)

            if diagnosis:
                if support:
                    st.success(f"✅ The X-ray supports the diagnosis of **{diagnosis}**.")
                else:
                    st.warning(f"⚠️ The X-ray does **not clearly support** the diagnosis of **{diagnosis}**.")
