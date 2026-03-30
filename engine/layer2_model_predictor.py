from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    LAYER2_FEATURE_COLUMNS_FILE,
    LAYER2_LABEL_ENCODER_FILE,
    LAYER2_METADATA_FILE,
    LAYER2_PIPELINE_FILE,
    NON_MENSTRUAL_PHASES,
    SUPPORTED_SYMPTOMS,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

SYMPTOM_ALIASES = {
    "headache": "headaches",
    "headaches": "headaches",
    "cramp": "cramps",
    "cramps": "cramps",
    "sorebreast": "sorebreasts",
    "sorebreasts": "sorebreasts",
    "fatigue": "fatigue",
    "sleepissue": "sleepissue",
    "sleepissues": "sleepissue",
    "sleep_issue": "sleepissue",
    "moodswing": "moodswing",
    "moodswings": "moodswing",
    "stress": "stress",
    "foodcraving": "foodcravings",
    "foodcravings": "foodcravings",
    "indigestion": "indigestion",
    "bloating": "bloating",
}


MUCUS_FERTILITY_MAP = {
    "unknown": 0.0,
    "dry": 0.0,
    "sticky": 0.25,
    "creamy": 0.50,
    "watery": 0.85,
    "eggwhite": 1.00,
}


def _safe_int(value, default=0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _normalize_symptom_name(symptom: str) -> Optional[str]:
    if symptom is None:
        return None
    key = str(symptom).strip().lower().replace(" ", "").replace("-", "").replace("/", "")
    return SYMPTOM_ALIASES.get(key)


def _normalize_symptom_list(symptoms: Optional[List[str]]) -> List[str]:
    out = []
    for s in symptoms or []:
        norm = _normalize_symptom_name(s)
        if norm and norm in SUPPORTED_SYMPTOMS and norm not in out:
            out.append(norm)
    return out


def _normalize_mucus(mucus: Optional[str]) -> str:
    if mucus is None:
        return "unknown"
    m = str(mucus).strip().lower()
    return m if m in MUCUS_FERTILITY_MAP else "unknown"


def _build_today_row(
    symptoms: Optional[List[str]],
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
) -> Dict[str, object]:
    symptom_set = set(_normalize_symptom_list(symptoms))
    mucus_type = _normalize_mucus(cervical_mucus)

    row: Dict[str, object] = {}

    # raw symptom values + logged flags
    for sym in SUPPORTED_SYMPTOMS:
        row[sym] = 1.0 if sym in symptom_set else 0.0
        row[f"{sym}_logged"] = 1

    # grouped features
    pain_cols = ["headaches", "cramps", "sorebreasts"]
    energy_cols = ["fatigue", "sleepissue"]
    mood_cols = ["moodswing", "stress"]
    digestive_cols = ["foodcravings", "indigestion", "bloating"]

    def add_group(group_name: str, cols: List[str]) -> None:
        values = [float(row[c]) for c in cols]
        row[f"{group_name}_mean"] = float(np.mean(values)) if values else 0.0
        row[f"{group_name}_max"] = float(np.max(values)) if values else 0.0
        row[f"{group_name}_logged_count"] = len(cols)
        row[f"{group_name}_missing_frac"] = 0.0

    add_group("pain", pain_cols)
    add_group("energy", energy_cols)
    add_group("mood", mood_cols)
    add_group("digestive", digestive_cols)

    # completeness
    row["num_symptoms_logged"] = len(SUPPORTED_SYMPTOMS)
    row["symptom_completeness"] = 1.0

    # mucus
    row["mucus_logged"] = 0 if mucus_type == "unknown" else 1
    row["mucus_score_logged"] = 0 if mucus_type == "unknown" else 1
    row["mucus_fertility_score"] = float(MUCUS_FERTILITY_MAP[mucus_type])

    # simple user inputs
    row["appetite"] = _safe_int(appetite, 0)
    row["exerciselevel"] = _safe_int(exerciselevel, 0)
    row["mucus_type"] = mucus_type

    return row


def _build_recent_rows(
    recent_daily_logs: Optional[List[Dict[str, object]]],
    today_row: Dict[str, object],
) -> List[Dict[str, object]]:
    """
    Expects up to 2 prior daily logs, oldest -> newest.
    Each prior log can contain:
      {
        "symptoms": [...],
        "cervical_mucus": "creamy",
        "appetite": 1,
        "exerciselevel": 2
      }
    """
    rows: List[Dict[str, object]] = []

    for item in recent_daily_logs or []:
        rows.append(
            _build_today_row(
                symptoms=item.get("symptoms", []),
                cervical_mucus=item.get("cervical_mucus", "unknown"),
                appetite=item.get("appetite", 0),
                exerciselevel=item.get("exerciselevel", 0),
            )
        )

    rows.append(today_row)
    return rows[-3:]  # keep only last 3 total


def _apply_history_features(last_rows: List[Dict[str, object]], today_row: Dict[str, object]) -> Dict[str, object]:
    """
    Recreates the main lag / rolling features expected by the new model.
    """
    out = dict(today_row)

    history_base_cols = [
        "pain_mean", "pain_max",
        "energy_mean", "energy_max",
        "mood_mean", "mood_max",
        "digestive_mean", "digestive_max",
        "mucus_fertility_score",
        "num_symptoms_logged",
        "symptom_completeness",
    ]

    rows = last_rows[-3:]
    n = len(rows)

    def val(r_idx: int, col: str) -> float:
        if r_idx < 0 or r_idx >= n:
            return 0.0
        try:
            v = rows[r_idx].get(col, 0.0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.0
            return float(v)
        except Exception:
            return 0.0

    for col in history_base_cols:
        current = val(n - 1, col)
        lag1 = val(n - 2, col) if n >= 2 else 0.0
        lag2 = val(n - 3, col) if n >= 3 else 0.0
        series = [val(i, col) for i in range(n)]

        out[f"{col}_lag1"] = lag1
        out[f"{col}_lag2"] = lag2
        out[f"{col}_roll3_mean"] = float(np.mean(series)) if series else current
        out[f"{col}_roll3_max"] = float(np.max(series)) if series else current
        out[f"{col}_trend2"] = float(current - lag2)
        out[f"{col}_persist3"] = float(sum(1 for x in series if x > 0))

    return out


def _make_feature_frame(
    symptoms: Optional[List[str]],
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
) -> pd.DataFrame:
    today_row = _build_today_row(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
    )
    history_rows = _build_recent_rows(recent_daily_logs=recent_daily_logs, today_row=today_row)
    final_row = _apply_history_features(last_rows=history_rows, today_row=today_row)

    feature_cols = joblib.load(ARTIFACTS_DIR / LAYER2_FEATURE_COLUMNS_FILE)

    X = pd.DataFrame([{col: final_row.get(col, np.nan) for col in feature_cols}])

    # fill any missing categoricals the pipeline may expect
    for c in X.columns:
        if X[c].dtype == "object" and X[c].isna().all():
            X[c] = "unknown"

    return X


def _get_signal_confidence(phase_probs: Dict[str, float], symptom_count: int, cervical_mucus: str) -> str:
    sorted_probs = sorted(phase_probs.values(), reverse=True)
    top_prob = sorted_probs[0] if sorted_probs else 0.0
    second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    gap = top_prob - second_prob
    mucus = _normalize_mucus(cervical_mucus)

    if gap >= 0.30 and (symptom_count >= 2 or mucus in {"watery", "eggwhite"}):
        return "high"
    if gap >= 0.15 and (symptom_count >= 1 or mucus != "unknown"):
        return "medium"
    return "low"


def _get_fertility_status(phase_probs: Dict[str, float], cervical_mucus: str, symptom_count: int) -> str:
    fertility_prob = phase_probs.get("Fertility", 0.0)
    mucus = _normalize_mucus(cervical_mucus)

    if mucus in {"watery", "eggwhite"} and fertility_prob >= 0.40:
        return "Red Day"
    if fertility_prob >= 0.60:
        return "Red Day"
    if fertility_prob >= 0.35 or mucus in {"creamy", "watery"}:
        return "Light Red Day"
    if symptom_count == 0 and mucus == "unknown":
        return "Need More Data"
    return "Green Day"


def _build_explanations(
    symptoms: Optional[List[str]],
    cervical_mucus: str,
    top_phase: str,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
) -> List[str]:
    symptom_set = set(_normalize_symptom_list(symptoms))
    mucus = _normalize_mucus(cervical_mucus)
    explanations: List[str] = []

    if mucus in {"watery", "eggwhite"}:
        explanations.append("Fertile-type cervical mucus increased fertility likelihood.")
    elif mucus == "creamy":
        explanations.append("Creamy cervical mucus supported a possible fertility transition.")
    elif mucus in {"dry", "sticky"}:
        explanations.append("Dry or sticky mucus lowered fertile-window likelihood.")

    if top_phase == "Luteal" and {"sorebreasts", "foodcravings", "bloating"} & symptom_set:
        explanations.append("Breast tenderness, cravings, or bloating supported a luteal-like pattern.")

    if top_phase == "Follicular" and len(symptom_set) <= 2 and mucus in {"unknown", "sticky", "creamy", "dry"}:
        explanations.append("Lower or less specific symptom burden fit better with a follicular-like pattern.")

    if top_phase == "Fertility" and mucus in {"watery", "eggwhite", "creamy"}:
        explanations.append("Body signals matched a more fertile phase pattern.")

    if recent_daily_logs:
        explanations.append("Recent 3-day symptom history was used to stabilize the phase estimate.")

    if not explanations:
        explanations.append("Current symptom pattern was used to refine today’s non-menstrual phase estimate.")

    return explanations[:3]


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_layer2_output(
    symptoms: Optional[List[str]],
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    Layer 2 predicts ONLY:
      - Follicular
      - Fertility
      - Luteal

    Menstrual is handled later by app logic when Period Start is logged.
    """
    pipeline = joblib.load(ARTIFACTS_DIR / LAYER2_PIPELINE_FILE)
    label_encoder = joblib.load(ARTIFACTS_DIR / LAYER2_LABEL_ENCODER_FILE)

    X = _make_feature_frame(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
        recent_daily_logs=recent_daily_logs,
    )

    probs = pipeline.predict_proba(X)[0]
    classes_encoded = pipeline.classes_
    class_names = label_encoder.inverse_transform(classes_encoded)

    phase_probs = {phase: float(prob) for phase, prob in zip(class_names, probs)}

    # make sure all 3 non-menstrual phases are present
    for phase in NON_MENSTRUAL_PHASES:
        phase_probs.setdefault(phase, 0.0)

    # normalize just in case
    total = sum(phase_probs.values()) or 1.0
    phase_probs = {k: v / total for k, v in phase_probs.items()}

    top_phase = max(phase_probs, key=phase_probs.get)
    symptom_count = len(_normalize_symptom_list(symptoms))

    sorted_items = sorted(phase_probs.items(), key=lambda x: x[1], reverse=True)
    top_prob = sorted_items[0][1]
    second_prob = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    prob_gap = top_prob - second_prob

    return {
        "phase_probs": phase_probs,
        "top_phase": top_phase,
        "top_prob": float(top_prob),
        "second_prob": float(second_prob),
        "prob_gap": float(prob_gap),
        "fertility_status": _get_fertility_status(
            phase_probs=phase_probs,
            cervical_mucus=cervical_mucus,
            symptom_count=symptom_count,
        ),
        "signal_confidence": _get_signal_confidence(
            phase_probs=phase_probs,
            symptom_count=symptom_count,
            cervical_mucus=cervical_mucus,
        ),
        "explanations": _build_explanations(
            symptoms=symptoms,
            cervical_mucus=cervical_mucus,
            top_phase=top_phase,
            recent_daily_logs=recent_daily_logs,
        ),
        "features_used": X.to_dict(orient="records")[0],
    }
