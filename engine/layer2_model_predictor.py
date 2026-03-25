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


def build_base_flags(symptoms: List[str], appetite: int = 0, exerciselevel: int = 0) -> Dict[str, int]:
    symptom_set = set(symptoms or [])
    row = {sym: int(sym in symptom_set) for sym in SUPPORTED_SYMPTOMS}
    row["appetite"] = int(appetite)
    row["exerciselevel"] = int(exerciselevel)
    return row


def build_engineered_features(
    symptoms: List[str],
    cervical_mucus: str,
    appetite: int = 0,
    exerciselevel: int = 0,
) -> pd.DataFrame:
    artifacts = joblib.load(ARTIFACTS_DIR / "layer2a_artifacts.joblib")
    feature_cols = artifacts["feature_cols"]
    pca = artifacts["pca"]
    pca_input_cols = artifacts["pca_input_cols"]

    row = build_base_flags(
        symptoms=symptoms,
        appetite=appetite,
        exerciselevel=exerciselevel,
    )

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

    for i in range(4):
        row[f"PC{i+1}"] = float(pcs[0, i])

    X = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])
    return X


def get_fertility_status(
    phase_probs: Dict[str, float],
    cervical_mucus: str,
    symptom_count: int,
) -> str:
    fertility_prob = phase_probs.get("Fertility", 0.0)
    mucus = (cervical_mucus or "unknown").lower()

    if mucus in {"watery", "eggwhite"} and fertility_prob >= 0.45:
        return "Red Day"

    if fertility_prob >= 0.55:
        return "Red Day"

    if fertility_prob >= 0.35 or mucus in {"creamy", "watery"}:
        return "Light Red Day"

    if symptom_count == 0 and mucus == "unknown":
        return "Need More Data"

    return "Green Day"


def get_signal_confidence(phase_probs: Dict[str, float], symptom_count: int, cervical_mucus: str) -> str:
    sorted_probs = sorted(phase_probs.values(), reverse=True)
    gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    mucus = (cervical_mucus or "unknown").lower()

    if gap >= 0.35 and (symptom_count >= 2 or mucus in {"watery", "eggwhite"}):
        return "high"
    if gap >= 0.18 and (symptom_count >= 1 or mucus != "unknown"):
        return "medium"
    return "low"


def build_explanations(symptoms: List[str], cervical_mucus: str, top_phase: str) -> List[str]:
    explanations = []
    symptom_set = set(symptoms or [])
    mucus = (cervical_mucus or "unknown").lower()

    if mucus in {"watery", "eggwhite"}:
        explanations.append("Fertile-type cervical mucus increased fertility likelihood.")
    elif mucus == "creamy":
        explanations.append("Creamy cervical mucus supported a possible fertility transition.")
    elif mucus in {"dry", "sticky"}:
        explanations.append("Dry or sticky mucus lowered fertile-window likelihood.")

    if top_phase == "Menstrual" and {"cramps", "fatigue"} & symptom_set:
        explanations.append("Cramps and fatigue supported a menstrual-like pattern.")

    if top_phase == "Luteal" and {"sorebreasts", "foodcravings", "bloating"} & symptom_set:
        explanations.append("Breast tenderness, cravings, or bloating supported a luteal-like pattern.")

    if top_phase == "Follicular" and len(symptom_set) <= 2 and mucus in {"unknown", "sticky", "creamy"}:
        explanations.append("Lower symptom burden fit better with a follicular-like pattern.")

    if top_phase == "Fertility" and mucus in {"watery", "eggwhite"}:
        explanations.append("Body signals matched a more fertile phase pattern.")

    if not explanations:
        explanations.append("Current symptom pattern was used to refine today’s phase estimate.")

    return explanations[:3]


def get_layer2_output(
    symptoms: List[str],
    cervical_mucus: str,
    appetite: int = 0,
    exerciselevel: int = 0,
) -> Dict[str, object]:
    model = joblib.load(ARTIFACTS_DIR / "layer2a_clf_mucus.joblib")
    X = build_engineered_features(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
    )

    probs = model.predict_proba(X)[0]
    classes = list(model.classes_)

    phase_probs = {phase: float(prob) for phase, prob in zip(classes, probs)}
    top_phase = max(phase_probs, key=phase_probs.get)
    symptom_count = len(symptoms or [])

    fertility_status = get_fertility_status(
        phase_probs=phase_probs,
        cervical_mucus=cervical_mucus,
        symptom_count=symptom_count,
    )

    signal_confidence = get_signal_confidence(
        phase_probs=phase_probs,
        symptom_count=symptom_count,
        cervical_mucus=cervical_mucus,
    )

    explanations = build_explanations(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        top_phase=top_phase,
    )

    return {
        "phase_probs": phase_probs,
        "top_phase": top_phase,
        "fertility_status": fertility_status,
        "signal_confidence": signal_confidence,
        "explanations": explanations,
        "features_used": X.to_dict(orient="records")[0],
    }
