from typing import Dict, List

import joblib
import pandas as pd

from .config import ARTIFACTS_DIR, SUPPORTED_SYMPTOMS


def mucus_to_score(mucus: str) -> int:
    mapping = {
        "dry": 0,
        "sticky": 1,
        "creamy": 2,
        "watery": 3,
        "eggwhite": 4,
        "unknown": 0,
    }
    return mapping.get((mucus or "unknown").lower(), 0)


def mucus_to_fertile_flag(mucus: str) -> int:
    return int((mucus or "").lower() in {"watery", "eggwhite"})


def build_base_flags(symptoms: List[str]) -> Dict[str, int]:
    symptom_set = set(symptoms or [])
    return {sym: int(sym in symptom_set) for sym in SUPPORTED_SYMPTOMS}


def build_engineered_features(symptoms: List[str], cervical_mucus: str) -> pd.DataFrame:
    artifacts = joblib.load(ARTIFACTS_DIR / "layer2a_artifacts.joblib")
    feature_cols = artifacts["feature_cols"]
    pca = artifacts["pca"]
    pca_input_cols = artifacts["pca_input_cols"]

    row = build_base_flags(symptoms)

    row["symptom_burden_score"] = sum(row[s] for s in SUPPORTED_SYMPTOMS)
    row["pain_score"] = row["cramps"] + row["headaches"]
    row["recovery_score"] = row["fatigue"] + row["sleepissue"]
    row["mood_score"] = row["moodswing"] + row["stress"] + row["foodcravings"]
    row["body_score"] = row["sorebreasts"] + row["bloating"]
    row["digestive_score"] = row["indigestion"] + row["bloating"]

    row["cramps__x__bloating"] = row["cramps"] * row["bloating"]
    row["sorebreasts__x__foodcravings"] = row["sorebreasts"] * row["foodcravings"]
    row["fatigue__x__sleepissue"] = row["fatigue"] * row["sleepissue"]
    row["stress__x__moodswing"] = row["stress"] * row["moodswing"]
    row["pain_score__x__body_score"] = row["pain_score"] * row["body_score"]
    row["digestive_score__x__mood_score"] = row["digestive_score"] * row["mood_score"]

    row["cervical_mucus_score"] = mucus_to_score(cervical_mucus)
    row["mucus_fertile_flag"] = mucus_to_fertile_flag(cervical_mucus)

    pca_input_df = pd.DataFrame([{col: row.get(col, 0) for col in pca_input_cols}])
    pcs = pca.transform(pca_input_df)

    for i in range(pcs.shape[1]):
        row[f"PC{i+1}"] = float(pcs[0, i])

    X = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])
    return X


def get_layer2_output(symptoms: List[str], cervical_mucus: str) -> Dict[str, object]:
    model = joblib.load(ARTIFACTS_DIR / "layer2a_clf_mucus.joblib")
    X = build_engineered_features(symptoms, cervical_mucus)

    probs = model.predict_proba(X)[0]
    classes = list(model.classes_)

    phase_probs = {phase: float(prob) for phase, prob in zip(classes, probs)}
    top_phase = max(phase_probs, key=phase_probs.get)

    return {
        "phase_probs": phase_probs,
        "top_phase": top_phase,
        "features_used": X.to_dict(orient="records")[0],
    }
