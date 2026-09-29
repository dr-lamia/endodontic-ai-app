from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PULPAL = [
    "Normal pulp",
    "Reversible pulpitis",
    "Symptomatic irreversible pulpitis",
    "Asymptomatic irreversible pulpitis",
    "Pulp necrosis",
]

APICAL = [
    "Normal apical tissues",
    "Symptomatic apical periodontitis",
    "Asymptomatic apical periodontitis",
    "Acute apical abscess",
    "Chronic apical abscess",
]

# Expert-informed STARTING assumptions for a research prototype.
# These probabilities must be reviewed and approved by endodontic experts
# before the prospective validation model is formally frozen.
PULPAL_PRIOR = np.array([0.16, 0.20, 0.25, 0.14, 0.25])

APICAL_GIVEN_PULP = {
    "Normal pulp": np.array([0.94, 0.04, 0.015, 0.002, 0.003]),
    "Reversible pulpitis": np.array([0.86, 0.11, 0.02, 0.004, 0.006]),
    "Symptomatic irreversible pulpitis": np.array([0.45, 0.43, 0.07, 0.035, 0.015]),
    "Asymptomatic irreversible pulpitis": np.array([0.58, 0.14, 0.20, 0.02, 0.06]),
    "Pulp necrosis": np.array([0.05, 0.22, 0.34, 0.22, 0.17]),
}

PULP_FEATURE_P = {
    "Normal pulp": {
        "spontaneous_pain": 0.03, "cold_sensitivity": 0.08, "heat_sensitivity": 0.03,
        "lingering_thermal_pain": 0.01, "no_response_vitality": 0.02, "deep_caries": 0.08,
        "tooth_discoloration": 0.03, "previous_restoration": 0.30,
    },
    "Reversible pulpitis": {
        "spontaneous_pain": 0.10, "cold_sensitivity": 0.78, "heat_sensitivity": 0.15,
        "lingering_thermal_pain": 0.12, "no_response_vitality": 0.02, "deep_caries": 0.65,
        "tooth_discoloration": 0.04, "previous_restoration": 0.44,
    },
    "Symptomatic irreversible pulpitis": {
        "spontaneous_pain": 0.82, "cold_sensitivity": 0.74, "heat_sensitivity": 0.48,
        "lingering_thermal_pain": 0.86, "no_response_vitality": 0.03, "deep_caries": 0.72,
        "tooth_discoloration": 0.06, "previous_restoration": 0.42,
    },
    "Asymptomatic irreversible pulpitis": {
        "spontaneous_pain": 0.06, "cold_sensitivity": 0.38, "heat_sensitivity": 0.10,
        "lingering_thermal_pain": 0.13, "no_response_vitality": 0.04, "deep_caries": 0.80,
        "tooth_discoloration": 0.05, "previous_restoration": 0.45,
    },
    "Pulp necrosis": {
        "spontaneous_pain": 0.18, "cold_sensitivity": 0.04, "heat_sensitivity": 0.05,
        "lingering_thermal_pain": 0.02, "no_response_vitality": 0.93, "deep_caries": 0.55,
        "tooth_discoloration": 0.38, "previous_restoration": 0.48,
    },
}

APICAL_FEATURE_P = {
    "Normal apical tissues": {
        "pain_on_biting": 0.05, "percussion_sensitivity": 0.04, "palpation_sensitivity": 0.02,
        "swelling": 0.003, "sinus_tract": 0.002, "periapical_radiolucency": 0.01,
        "pdl_widening": 0.05, "mobility": 0.03,
    },
    "Symptomatic apical periodontitis": {
        "pain_on_biting": 0.81, "percussion_sensitivity": 0.88, "palpation_sensitivity": 0.48,
        "swelling": 0.06, "sinus_tract": 0.02, "periapical_radiolucency": 0.33,
        "pdl_widening": 0.59, "mobility": 0.11,
    },
    "Asymptomatic apical periodontitis": {
        "pain_on_biting": 0.08, "percussion_sensitivity": 0.11, "palpation_sensitivity": 0.06,
        "swelling": 0.02, "sinus_tract": 0.08, "periapical_radiolucency": 0.91,
        "pdl_widening": 0.45, "mobility": 0.08,
    },
    "Acute apical abscess": {
        "pain_on_biting": 0.88, "percussion_sensitivity": 0.92, "palpation_sensitivity": 0.84,
        "swelling": 0.91, "sinus_tract": 0.06, "periapical_radiolucency": 0.56,
        "pdl_widening": 0.62, "mobility": 0.24,
    },
    "Chronic apical abscess": {
        "pain_on_biting": 0.16, "percussion_sensitivity": 0.23, "palpation_sensitivity": 0.13,
        "swelling": 0.10, "sinus_tract": 0.91, "periapical_radiolucency": 0.89,
        "pdl_widening": 0.54, "mobility": 0.12,
    },
}

TOOTH_TYPES = ["Anterior", "Premolar", "Molar"]
TOOTH_PRIOR = [0.22, 0.28, 0.50]


def bernoulli(rng: np.random.Generator, p: float) -> int:
    return int(rng.random() < np.clip(p, 0.001, 0.999))


def generate(n: int, seed: int = 42, noise: float = 0.035) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for idx in range(1, n + 1):
        pulp = rng.choice(PULPAL, p=PULPAL_PRIOR)
        apical = rng.choice(APICAL, p=APICAL_GIVEN_PULP[pulp])

        row = {
            "synthetic_id": f"SYN-{idx:06d}",
            "age_years": int(np.clip(np.rint(rng.normal(39, 15)), 12, 85)),
            "tooth_type": rng.choice(TOOTH_TYPES, p=TOOTH_PRIOR),
        }

        for feature, p in PULP_FEATURE_P[pulp].items():
            row[feature] = bernoulli(rng, np.clip(rng.normal(p, noise), 0.005, 0.995))

        for feature, p in APICAL_FEATURE_P[apical].items():
            row[feature] = bernoulli(rng, np.clip(rng.normal(p, noise), 0.005, 0.995))

        row["systemic_signs"] = bernoulli(
            rng, 0.12 if apical == "Acute apical abscess" else 0.005
        )
        row["tooth_unrestorable"] = bernoulli(
            rng, 0.07 + (0.08 if row["deep_caries"] else 0.0)
        )

        if pulp == "Pulp necrosis" and row["no_response_vitality"] == 0 and rng.random() < 0.80:
            row["no_response_vitality"] = 1
        if apical == "Acute apical abscess" and row["swelling"] == 0 and rng.random() < 0.75:
            row["swelling"] = 1
        if apical == "Chronic apical abscess" and row["sinus_tract"] == 0 and rng.random() < 0.75:
            row["sinus_tract"] = 1

        if row["no_response_vitality"]:
            cold_test = "No response"
        elif row["lingering_thermal_pain"]:
            cold_test = "Exaggerated/lingering"
        elif row["cold_sensitivity"]:
            cold_test = "Positive, non-lingering"
        else:
            cold_test = "Normal/negative symptoms"

        row["cold_test_pattern"] = cold_test
        row["pulpal_diagnosis"] = pulp
        row["apical_diagnosis"] = apical
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_endodontic_v2.csv.gz"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(args.output, index=False, compression="gzip")
    print(f"Saved {len(df):,} synthetic records to {args.output}")
    print("Pulpal distribution:\n", df["pulpal_diagnosis"].value_counts(normalize=True).round(3))
    print("Apical distribution:\n", df["apical_diagnosis"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
