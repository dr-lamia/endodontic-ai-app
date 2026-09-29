from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import streamlit as st

from clinical_logic import proposed_management

MODEL_PATH = Path("models/endodontic_synthetic_v2.joblib")

st.set_page_config(page_title="Synthetic-Trained Endodontic AI — Validation V2", layout="wide")


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Frozen model not found at {MODEL_PATH}. Run generate_synthetic_v2.py and train_model_v2.py first."
        )
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
models = bundle["models"]
features = bundle["feature_columns"]

st.title("🦷 Synthetic-Trained Endodontic AI — Research Validation")
st.warning(
    "Research-use validation build. The AI output must not replace full clinical examination, radiographic assessment, "
    "or clinician judgment. Reference-standard experts should remain blinded to AI predictions."
)

with st.sidebar:
    st.header("Study case")
    case_id = st.text_input("De-identified study case ID", placeholder="REAL-0001")
    age_years = st.number_input("Age (years)", min_value=12, max_value=100, value=35, step=1)
    tooth_type = st.selectbox("Tooth type", ["Anterior", "Premolar", "Molar"])

    st.header("Symptoms")
    spontaneous_pain = st.checkbox("Spontaneous pain")
    pain_on_biting = st.checkbox("Pain on biting")
    cold_sensitivity = st.checkbox("Cold sensitivity")
    heat_sensitivity = st.checkbox("Heat sensitivity")
    lingering_thermal_pain = st.checkbox("Lingering thermal pain")

    st.header("Clinical examination")
    percussion_sensitivity = st.checkbox("Percussion sensitivity")
    palpation_sensitivity = st.checkbox("Palpation sensitivity")
    swelling = st.checkbox("Swelling")
    sinus_tract = st.checkbox("Sinus tract")
    mobility = st.checkbox("Mobility")
    deep_caries = st.checkbox("Deep caries")
    previous_restoration = st.checkbox("Previous/large restoration")
    tooth_discoloration = st.checkbox("Tooth discoloration")

    st.header("Pulp testing")
    no_response_vitality = st.checkbox("No response to vitality testing")

    st.header("Radiographic findings")
    periapical_radiolucency = st.checkbox("Periapical radiolucency")
    pdl_widening = st.checkbox("PDL widening")

    st.header("Management modifiers")
    systemic_signs = st.checkbox("Systemic signs / spreading infection")
    tooth_unrestorable = st.checkbox("Tooth considered unrestorable")

if no_response_vitality:
    cold_test_pattern = "No response"
elif lingering_thermal_pain:
    cold_test_pattern = "Exaggerated/lingering"
elif cold_sensitivity:
    cold_test_pattern = "Positive, non-lingering"
else:
    cold_test_pattern = "Normal/negative symptoms"

row = {
    "age_years": int(age_years),
    "tooth_type": tooth_type,
    "spontaneous_pain": int(spontaneous_pain),
    "cold_sensitivity": int(cold_sensitivity),
    "heat_sensitivity": int(heat_sensitivity),
    "lingering_thermal_pain": int(lingering_thermal_pain),
    "no_response_vitality": int(no_response_vitality),
    "deep_caries": int(deep_caries),
    "tooth_discoloration": int(tooth_discoloration),
    "previous_restoration": int(previous_restoration),
    "pain_on_biting": int(pain_on_biting),
    "percussion_sensitivity": int(percussion_sensitivity),
    "palpation_sensitivity": int(palpation_sensitivity),
    "swelling": int(swelling),
    "sinus_tract": int(sinus_tract),
    "periapical_radiolucency": int(periapical_radiolucency),
    "pdl_widening": int(pdl_widening),
    "mobility": int(mobility),
    "systemic_signs": int(systemic_signs),
    "tooth_unrestorable": int(tooth_unrestorable),
    "cold_test_pattern": cold_test_pattern,
}
input_df = pd.DataFrame([row])[features]

st.subheader("Real-patient checklist")
st.dataframe(pd.DataFrame({"Variable": list(row.keys()), "Value": list(row.values())}), use_container_width=True)

if st.button("🔒 Lock case and run frozen synthetic-trained model", type="primary"):
    if not case_id.strip():
        st.error("Enter a de-identified study case ID before generating a prediction.")
    else:
        pulp_model = models["pulpal_diagnosis"]
        apical_model = models["apical_diagnosis"]

        pulp_pred = pulp_model.predict(input_df)[0]
        apical_pred = apical_model.predict(input_df)[0]

        pulp_proba = pulp_model.predict_proba(input_df)[0]
        apical_proba = apical_model.predict_proba(input_df)[0]

        plan = proposed_management(
            pulp_pred, apical_pred, int(systemic_signs), int(tooth_unrestorable)
        )

        st.session_state["prediction"] = {
            "case_id": case_id.strip(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **row,
            "ai_pulpal_diagnosis": pulp_pred,
            "ai_pulpal_model_probability": round(float(max(pulp_proba)), 6),
            "ai_apical_diagnosis": apical_pred,
            "ai_apical_model_probability": round(float(max(apical_proba)), 6),
            "ai_management_pathway": plan,
        }

if "prediction" in st.session_state:
    pred = st.session_state["prediction"]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted pulpal diagnosis", pred["ai_pulpal_diagnosis"])
        st.caption(
            f"Model probability: {pred['ai_pulpal_model_probability']:.1%} "
            "(not yet clinically calibrated)"
        )
    with c2:
        st.metric("Predicted apical diagnosis", pred["ai_apical_diagnosis"])
        st.caption(
            f"Model probability: {pred['ai_apical_model_probability']:.1%} "
            "(not yet clinically calibrated)"
        )

    st.subheader("Proposed management pathway")
    st.info(pred["ai_management_pathway"])

    out = pd.DataFrame([pred])
    st.download_button(
        "Download case prediction CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"{pred['case_id']}_ai_prediction.csv",
        mime="text/csv",
    )

    with st.expander("Important validation rule"):
        st.write(
            "The expert-consensus diagnosis should be recorded independently of this AI output. "
            "Do not retrain, tune, recalibrate, or alter the model using any patient from the final validation cohort."
        )
