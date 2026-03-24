from __future__ import annotations

from datetime import date
from typing import Dict, List


PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

SUPPORTED_SYMPTOMS = [
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
]

MUCUS_OPTIONS = ["unknown", "dry", "sticky", "creamy", "watery", "eggwhite"]


def _normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in probs.values())
    if total <= 0:
        return {p: 1.0 / len(PHASES) for p in PHASES}
    return {p: max(v, 0.0) / total for p, v in probs.items()}


def _confidence_from_probs(probs: Dict[str, float]) -> float:
    vals = sorted(probs.values(), reverse=True)
    top1 = vals[0]
    top2 = vals[1] if len(vals) > 1 else 0.0
    return round(top1 - top2, 4)


def get_layer2_output(
    log_date: date,
    last_period_start: date,
    cycle_day: int,
    cycle_length: int,
    symptoms: List[str],
    cervical_mucus: str,
) -> Dict[str, object]:
    """
    Stable rule-based Layer 2A based on your final insight:
    - mucus is strongest
    - symptoms are supportive
    - luteal / menstrual signals can shift prediction
    """
    symptoms = [s for s in symptoms if s in SUPPORTED_SYMPTOMS]
    mucus = cervical_mucus if cervical_mucus in MUCUS_OPTIONS else "unknown"

    score = {
        "Menstrual": 0.05,
        "Follicular": 0.08,
        "Fertility": 0.08,
        "Luteal": 0.08,
    }

    has = lambda x: x in symptoms

    # -------- Menstrual-like evidence --------
    if has("cramps"):
        score["Menstrual"] += 0.28

    if has("cramps") and has("bloating"):
        score["Menstrual"] += 0.12

    if has("fatigue"):
        score["Menstrual"] += 0.05

    if has("headaches"):
        score["Menstrual"] += 0.05

    # -------- Luteal-like evidence --------
    if has("sorebreasts"):
        score["Luteal"] += 0.20

    if has("foodcravings"):
        score["Luteal"] += 0.12

    if has("moodswing"):
        score["Luteal"] += 0.10

    if has("sleepissue"):
        score["Luteal"] += 0.08

    if has("fatigue"):
        score["Luteal"] += 0.08

    if has("stress"):
        score["Luteal"] += 0.05

    if has("indigestion"):
        score["Luteal"] += 0.04

    if has("bloating"):
        score["Luteal"] += 0.08

    # -------- Mucus: strongest signal --------
    if mucus == "dry":
        score["Follicular"] += 0.06

    elif mucus == "sticky":
        score["Follicular"] += 0.10

    elif mucus == "creamy":
        score["Follicular"] += 0.16
        score["Fertility"] += 0.08

    elif mucus == "watery":
        score["Fertility"] += 0.35
        score["Menstrual"] -= 0.04

    elif mucus == "eggwhite":
        score["Fertility"] += 0.48
        score["Menstrual"] -= 0.06

    probs = _normalize_probs(score)
    predicted_phase = max(probs, key=probs.get)
    confidence = _confidence_from_probs(probs)

    return {
        "log_date": log_date,
        "cycle_day": cycle_day,
        "cycle_length": cycle_length,
        "predicted_phase": predicted_phase,
        "confidence": confidence,
        "phase_probabilities": probs,
        "symptoms_used": symptoms,
        "cervical_mucus": mucus,
    }
