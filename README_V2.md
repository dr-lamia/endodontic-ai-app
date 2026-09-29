# Synthetic-Trained Endodontic AI — Validation V2

This research branch tests a specific hypothesis: whether an AI model trained exclusively on clinically structured synthetic tabular endodontic data can generalize to real patients.

## What is new in V2
- Separate pulpal and apical diagnostic targets.
- Probabilistic synthetic patient generator rather than a simple four-rule label generator.
- Large reproducible synthetic dataset generation.
- Comparison of multiple machine-learning algorithms.
- Saved model hash + dataset hash for model freezing.
- Real-patient checklist that uses exactly the same variables as the synthetic training schema.
- Prospective external-validation protocol with blinded expert consensus.
- Separate treatment-pathway and safety evaluation.
- Expert clinical/educational usefulness survey.

## Important
The current probability tables in the synthetic generator are a prototype. Endodontic experts must review/approve them before the final model is trained and frozen for prospective validation.

## Build the synthetic dataset

```bash
python generate_synthetic_v2.py --n 20000
```

## Train candidate models

```bash
python train_model_v2.py
```

This creates:
- `models/endodontic_synthetic_v2.joblib`
- `models/model_metadata.json`

## Run the research app

```bash
streamlit run app_v2.py
```

## Validation principle
No real-patient case from the primary validation cohort may be used for training, tuning, recalibration, threshold adjustment, or model selection.

Synthetic hold-out results are only development checks. The scientific test is performance on an untouched real-patient cohort against an independent expert-consensus reference standard.
