# Prospective External Validation Protocol — Synthetic-Trained Endodontic AI

## Primary research question
Can a machine-learning system trained exclusively on clinically structured synthetic tabular data generalize to previously unseen real endodontic patients for diagnostic decision support?

## Primary aim
Externally validate the frozen synthetic-trained model against an independent expert-consensus reference standard in real patients.

## Primary endpoint
Macro-averaged F1 score for pulpal diagnosis on the untouched real-patient validation cohort.

## Key secondary endpoints
- Overall accuracy and balanced accuracy for pulpal diagnosis.
- Class-specific sensitivity, specificity, PPV and NPV.
- Macro-F1 and class-specific performance for apical diagnosis.
- Cohen's kappa versus expert consensus.
- Calibration of model probabilities on real patients (reported, not used to recalibrate the frozen model during the primary analysis).
- Concordance of proposed management pathway with independent expert judgement.
- Frequency of potentially unsafe management recommendations.
- Expert-rated clinical usefulness, educational usefulness and usability.

## Study design
Prospective diagnostic external-validation study with a secondary expert usability/acceptability evaluation.

### Phase A — synthetic-only model development
1. Generate the pre-specified synthetic dataset from the expert-reviewed probabilistic generator.
2. Split synthetic data into development and synthetic hold-out subsets.
3. Compare candidate algorithms using only synthetic data.
4. Select the final model according to pre-specified macro-F1/balanced-accuracy criteria.
5. Save model artifact + SHA-256 hash + generator version + dataset hash.
6. Freeze the model before enrolment/analysis of the final real-patient cohort.

### Phase B — prospective real-patient validation
For each eligible real patient/tooth:
1. Assign a de-identified study case ID.
2. Record the exact checklist variables used by the model.
3. Run the frozen model without any training or updating.
4. Store pulpal diagnosis, apical diagnosis, model probabilities and proposed management pathway.
5. Independently establish the reference standard by two qualified endodontic experts using the full clinical/radiographic case information.
6. If experts disagree, use a pre-specified third-expert adjudication procedure.
7. Merge AI and reference-standard data only after expert ratings are completed.

## Blinding
Reference-standard experts should not see the AI prediction before submitting their diagnosis and treatment judgement. The clinician entering checklist data should not alter the recorded findings after viewing the AI output unless the correction is documented as a data-entry error.

## Proposed study population
Permanent teeth undergoing diagnostic assessment for suspected pulpal and/or apical disease.

### Initial inclusion criteria
- Real patients receiving endodontic diagnostic assessment.
- Permanent tooth.
- Complete history, clinical examination, pulp testing and required radiographic assessment sufficient for expert diagnosis.
- Patient consent/ethics approval as required by the study institution.

### Initial exclusion criteria
- Previously root-filled or previously initiated endodontic treatment for the index tooth, unless a separate retreatment model is developed.
- Primary teeth.
- Cases dominated by trauma, resorption, immature-apex management, or non-endodontic pain when the current model does not represent those conditions.
- Missing essential predictor data.
- Any diagnostic class not represented in the locked model should be recorded as out-of-scope rather than forced into an available class.

## Diagnostic scope for V2
Pulpal target classes:
- Normal pulp
- Reversible pulpitis
- Symptomatic irreversible pulpitis
- Asymptomatic irreversible pulpitis
- Pulp necrosis

Apical target classes:
- Normal apical tissues
- Symptomatic apical periodontitis
- Asymptomatic apical periodontitis
- Acute apical abscess
- Chronic apical abscess

The protocol intentionally uses established AAE terminology for this V2 build. AAE and ESE are actively reviewing a newer joint classification; the locked study terminology should not be changed after validation starts.

## Treatment-pathway evaluation
Treatment is evaluated separately from diagnostic accuracy because multiple management options may be clinically acceptable.

Each expert rates the AI management pathway as:
- 2 = appropriate/clinically acceptable
- 1 = partially appropriate; modification required
- 0 = inappropriate

A separate binary safety item is recorded:
- potentially unsafe recommendation: Yes / No

The management output is decision support, not an autonomous prescription.

## Real-patient sample-size target
Plan approximately 200–250 validation cases, then finalize sample size against the chosen primary metric and the expected class-specific precision. Recruitment should ensure adequate representation of each major diagnosis; overall sample size alone is insufficient if rare classes contain too few cases.

## Analysis plan
- Lock the model and analysis code before final outcome analysis.
- Report the patient/tooth flow diagram and exclusions.
- Provide confusion matrices for both pulpal and apical targets.
- Report accuracy, balanced accuracy, macro-F1, weighted-F1 and per-class precision/recall/F1.
- Compute one-vs-rest sensitivity, specificity, PPV and NPV with 95% confidence intervals.
- Report Cohen's kappa for AI versus expert consensus.
- For model probabilities, report Brier score and calibration plots where applicable.
- Analyze treatment ratings and unsafe-recommendation rate separately.
- Do not tune the model on the validation cohort in the primary analysis.

## Secondary expert evaluation
After using the system, experts complete a structured survey covering clinical usefulness, educational usefulness, clarity, trust, safety, workflow fit and usability. SUS can be added as a standardized usability measure.

## Core claim permitted if validation is successful
The study may support the statement that a model trained exclusively on clinically structured synthetic tabular data demonstrated measurable generalization to an independent cohort of real endodontic patients.

It should not claim that synthetic data replace real clinical evidence.
