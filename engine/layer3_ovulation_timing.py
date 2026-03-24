from datetime import datetime, timedelta
from typing import Dict

from .utils import parse_date


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _confidence_bucket(fertility_prob: float, shift_abs: float) -> str:
    if fertility_prob >= 0.65 and shift_abs <= 4:
        return "high"
    if fertility_prob >= 0.45:
        return "moderate"
    return "low"


def _consistency_label(shift_abs: float) -> str:
    if shift_abs <= 2:
        return "aligned"
    if shift_abs <= 6:
        return "mild_shift"
    return "strong_contradiction"


def get_layer3_output(layer1: Dict, layer2: Dict, cervical_mucus: str = "unknown") -> Dict:
    cycle_length = layer1.get("estimated_cycle_length")
    cycle_day = layer1.get("cycle_day")
    baseline_next_period = layer1.get("predicted_next_period")

    if cycle_length is None or cycle_day is None or baseline_next_period is None:
        return {
            "baseline_ovulation_day": None,
            "expected_symptom_day": None,
            "shift_days": None,
            "adjusted_cycle_day": None,
            "adjusted_ovulation_date": None,
            "fertile_window_start": None,
            "fertile_window_end": None,
            "adjusted_next_period_date": baseline_next_period,
            "consistency": "unknown",
            "timing_confidence": "low",
        }

    probs = layer2["phase_probs"]

    L = float(cycle_length)
    baseline_cycle_day = float(cycle_day)
    ovulation_anchor = max(round(L - 14), 1)
    menstrual_anchor = 3
    follicular_anchor = max(round((6 + ovulation_anchor) / 2), 6)
    luteal_anchor = min(round((ovulation_anchor + L) / 2), round(L))

    expected_symptom_day = (
        probs.get("Menstrual", 0.0) * menstrual_anchor +
        probs.get("Follicular", 0.0) * follicular_anchor +
        probs.get("Fertility", 0.0) * ovulation_anchor +
        probs.get("Luteal", 0.0) * luteal_anchor
    )

    shift_days = expected_symptom_day - baseline_cycle_day

    adjustment_strength = 0.35
    fertility_prob = probs.get("Fertility", 0.0)

    if fertility_prob >= 0.60:
        adjustment_strength += 0.15
    if (cervical_mucus or "").lower() in {"watery", "eggwhite"}:
        adjustment_strength += 0.15

    adjusted_cycle_day = baseline_cycle_day + adjustment_strength * shift_days
    adjusted_cycle_day = _clamp(adjusted_cycle_day, 1, L)

    today = datetime.today().date()
    adjusted_days_until_ovulation = ovulation_anchor - adjusted_cycle_day
    adjusted_days_until_next_period = L - adjusted_cycle_day

    adjusted_ovulation_date = today + timedelta(days=round(adjusted_days_until_ovulation))
    fertile_window_start = adjusted_ovulation_date - timedelta(days=2)
    fertile_window_end = adjusted_ovulation_date + timedelta(days=1)
    adjusted_next_period_date = today + timedelta(days=round(adjusted_days_until_next_period))

    return {
        "baseline_ovulation_day": ovulation_anchor,
        "expected_symptom_day": round(expected_symptom_day, 2),
        "shift_days": round(shift_days, 2),
        "adjusted_cycle_day": round(adjusted_cycle_day, 2),
        "adjusted_ovulation_date": adjusted_ovulation_date.isoformat(),
        "fertile_window_start": fertile_window_start.isoformat(),
        "fertile_window_end": fertile_window_end.isoformat(),
        "adjusted_next_period_date": adjusted_next_period_date.isoformat(),
        "consistency": _consistency_label(abs(shift_days)),
        "timing_confidence": _confidence_bucket(fertility_prob, abs(shift_days)),
    }
