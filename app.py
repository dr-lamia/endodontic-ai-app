import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Endodontic Multimodal AI Assistant", layout="wide")


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

# --- API Keys (set in Streamlit Cloud secrets) ---
HF_TOKEN = st.secrets.get("HF_TOKEN")

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
st.set_page_config(page_title="Endodontic Multimodal AI Assistant", layout="wide")
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

# --- 3. Vision-Language X-ray Analysis ---
st.markdown("---")
st.header("📷 X-ray Analysis with AI Vision-Language Models")

model_choice = st.selectbox("Choose Vision-Language Model:", [
    "BLIP", "ViT-GPT2", "BioViL-T", "ClinicalCamel", "OpenAI GPT-4 Vision", "Gemini Vision"
])

uploaded_file = st.file_uploader("Upload dental X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    processed = ImageOps.exif_transpose(image).resize((512, 512))
    st.image(processed, caption="Processed X-ray", use_column_width=True)
    image_bytes = io.BytesIO()
    processed.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    if model_choice in ["BLIP", "ViT-GPT2"]:
        hf_model = {
            "BLIP": "Salesforce/blip-image-captioning-base",
            "ViT-GPT2": "nlpconnect/vit-gpt2-image-captioning"
        }[model_choice]
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{hf_model}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            files={"inputs": image_bytes}
        )
        if response.ok:
            caption = response.json()[0].get("generated_text", "No caption.")
            st.success("🧠 Model Caption: " + caption)
            if "diagnosis" in st.session_state:
                st.markdown(f"**🔁 Correlation:** The caption above may indicate _{st.session_state['diagnosis']}_ if consistent with clinical symptoms.")
        else:
            st.error("❌ Failed to analyze image.")

    elif model_choice == "BioViL-T":
        st.info("BioViL-T returns visual embeddings for downstream diagnosis tasks.")

    elif model_choice == "ClinicalCamel":
        if "diagnosis" in st.session_state:
            st.info("📋 ClinicalCamel can cross-check diagnosis from symptoms.")
        else:
            st.warning("Run symptom-based diagnosis first.")

    elif model_choice == "OpenAI GPT-4 Vision":
        st.warning("Requires OpenAI API key & integration (not public).")

    elif model_choice == "Gemini Vision":
        st.warning("Requires Gemini API key (coming soon).")


# --- 4. Chat with Assistant (Simple Memory) ---
st.markdown("---")
st.header("💬 Chat with AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the diagnosis or treatment:")
if user_input:
    # Very simple echo logic (can be replaced with GPT API)
    diagnosis = st.session_state.get("diagnosis", "a dental condition")
    answer = f"As an AI assistant, I suggest consulting a dentist for detailed management of {diagnosis}. Your question was: '{user_input}'"
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", answer))

for sender, message in st.session_state.chat_history[-6:]:  # Show last 3 exchanges
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 AI:** {message}")


# --- 4. Chat with AI Assistant (Advanced with Save/Export) ---
import datetime

st.markdown("---")
st.header("💬 Chat with AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the diagnosis or treatment:")
if user_input:
    diagnosis = st.session_state.get("diagnosis", "a dental condition")

    # Placeholder AI logic (for GPT API, use OpenAI here)
    answer = f"As an AI assistant, I suggest consulting a dentist for detailed management of {diagnosis}. You asked: '{user_input}'"

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", answer))

# Display chat history
for sender, message in st.session_state.chat_history[-6:]:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 AI:** {message}")

# Save chat to .txt
if st.button("📝 Export Chat as .txt"):
    chat_lines = [f"{sender}: {message}" for sender, message in st.session_state.chat_history]
    chat_text = "\n".join(chat_lines)
    st.download_button("📥 Download Chat Log", chat_text, file_name="chat_history.txt")

# Save full case
if st.button("📁 Save Full Patient Case"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    diagnosis = st.session_state.get("diagnosis", "unknown")
    symptoms = [feat for feat, val in st.session_state.items() if feat in selected_features and val]
    chat_lines = [f"{sender}: {message}" for sender, message in st.session_state.chat_history]

    case_text = f"Patient Case - {timestamp}\nDiagnosis: {diagnosis}\nSymptoms: {', '.join(symptoms)}\n\nChat:\n" + "\n".join(chat_lines)
    st.download_button("📥 Download Full Case Log", case_text, file_name=f"patient_case_{timestamp}.txt")
