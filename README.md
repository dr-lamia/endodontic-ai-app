# 🦷 Endodontic Disease Diagnosis AI App

This Streamlit application uses AI to:
- Diagnose endodontic conditions based on symptoms
- Provide treatment suggestions
- Analyze X-ray images using the MedGemma vision-language model

## 🚀 Features
- Generates synthetic dataset and augments with SMOTE
- Selects top features with Random Forest
- Uses MedGemma to interpret dental X-rays
- Explains model predictions with feature importance

## 🧠 Diagnoses Supported
- Hyperemia
- Acute Pulpitis
- Chronic Pulpitis
- Periapical Abscess

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🖥️ Running the App
```bash
streamlit run app.py
```

## 🌐 Deployment on Streamlit Cloud
1. Upload to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Link your GitHub repository
4. Set `app.py` as the entry point

## 📂 Files Included
- `app.py` - Main Streamlit app
- `requirements.txt` - Dependencies
- `README.md` - Project documentation
