from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = ["pulpal_diagnosis", "apical_diagnosis"]
CATEGORICAL = ["tooth_type", "cold_test_pattern"]
NUMERIC = [
    "age_years", "spontaneous_pain", "cold_sensitivity", "heat_sensitivity",
    "lingering_thermal_pain", "no_response_vitality", "deep_caries",
    "tooth_discoloration", "previous_restoration", "pain_on_biting",
    "percussion_sensitivity", "palpation_sensitivity", "swelling", "sinus_tract",
    "periapical_radiolucency", "pdl_widening", "mobility", "systemic_signs",
    "tooth_unrestorable",
]
FEATURES = NUMERIC + CATEGORICAL


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])


def candidates(seed: int):
    return {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=450, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=450, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=250, random_state=seed
        ),
    }


def fit_one_target(df: pd.DataFrame, target: str, seed: int):
    X = df[FEATURES].copy()
    y = df[target].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

    scored = []
    fitted = {}
    for name, clf in candidates(seed).items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        scored.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
        })
        fitted[name] = (pipe, y_test, pred)

    scores = pd.DataFrame(scored).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"], ascending=False
    ).reset_index(drop=True)
    winner = scores.loc[0, "model"]
    winning_pipe, y_test, pred = fitted[winner]

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return winning_pipe, scores, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_endodontic_v2.csv.gz"))
    parser.add_argument("--outdir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    bundle = {
        "feature_columns": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "models": {},
    }
    metrics = {}

    for target in TARGETS:
        model, scores, report = fit_one_target(df, target, args.seed)
        bundle["models"][target] = model
        metrics[target] = {
            "candidate_scores": scores.to_dict(orient="records"),
            "classification_report": report,
            "selected_model": scores.loc[0, "model"],
        }
        print(f"\n{target}:\n{scores.to_string(index=False)}")

    model_path = args.outdir / "endodontic_synthetic_v2.joblib"
    joblib.dump(bundle, model_path, compress=3)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_source": "synthetic_only",
        "synthetic_generator_version": "2.0",
        "n_training_source_rows": int(len(df)),
        "random_seed": args.seed,
        "dataset_sha256": sha256(args.data),
        "model_sha256": sha256(model_path),
        "feature_columns": FEATURES,
        "targets": TARGETS,
        "metrics_on_synthetic_holdout": metrics,
        "IMPORTANT": (
            "Synthetic hold-out performance is development evidence only. The locked model must be "
            "evaluated on a completely untouched real-patient cohort before clinical claims are made."
        ),
    }
    with (args.outdir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model bundle to {model_path}")


if __name__ == "__main__":
    main()
