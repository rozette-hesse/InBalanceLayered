from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional


PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]


def _weighted_average(values: List[int]) -> Optional[float]:
    if not values:
        return None
    weights = list(range(1, len(values) + 1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def _normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in probs.values())
    if total <= 0:
        return {p: 1.0 / len(PHASES) for p in PHASES}
    return {p: max(v, 0.0) / total for p, v in probs.items()}


def _expected_phase_from_cycle_day(cycle_day: int, cycle_length: int = 28) -> str:
    if cycle_day <= 5:
        return "Menstrual"

    ovulation_day = max(12, round(cycle_length * 0.5))
    fertile_start = max(ovulation_day - 3, 6)
    fertile_end = ovulation_day + 1

    if fertile_start <= cycle_day <= fertile_end:
        return "Fertility"
    if cycle_day < fertile_start:
        return "Follicular"
    return "Luteal"


def _phase_prior_from_cycle_day(cycle_day: int, cycle_length: int) -> Dict[str, float]:
    phase = _expected_phase_from_cycle_day(cycle_day, cycle_length)

    priors = {
        "Menstrual": 0.10,
        "Follicular": 0.20,
        "Fertility": 0.20,
        "Luteal": 0.20,
    }

    if phase == "Menstrual":
        priors.update({
            "Menstrual": 0.60,
            "Follicular": 0.15,
            "Fertility": 0.05,
            "Luteal": 0.20,
        })
    elif phase == "Follicular":
        priors.update({
            "Menstrual": 0.05,
            "Follicular": 0.60,
            "Fertility": 0.20,
            "Luteal": 0.15,
        })
    elif phase == "Fertility":
        priors.update({
            "Menstrual": 0.02,
            "Follicular": 0.18,
            "Fertility": 0.62,
            "Luteal": 0.18,
        })
    elif phase == "Luteal":
        priors.update({
            "Menstrual": 0.10,
            "Follicular": 0.10,
            "Fertility": 0.05,
            "Luteal": 0.75,
        })

    return _normalize_probs(priors)


def get_layer1_output(period_starts: List[date], current_date: date) -> Dict[str, object]:
    """
    Stable Layer 1 output:
    - cycle lengths
    - estimated cycle length
    - baseline next period date
    - cycle day
    - phase prior
    """
    period_starts = sorted(period_starts)

    if len(period_starts) < 2:
        return {
            "cycle_lengths": [],
            "prediction": None,
            "phase_prior": None,
        }

    cycle_lengths = []
    for i in range(1, len(period_starts)):
        cycle_lengths.append((period_starts[i] - period_starts[i - 1]).days)

    estimated_cycle_length = round(_weighted_average(cycle_lengths))
    last_period_start = period_starts[-1]
    predicted_start = last_period_start + timedelta(days=estimated_cycle_length)

    cycle_day = max(1, (current_date - last_period_start).days + 1)
    likely_phase = _expected_phase_from_cycle_day(cycle_day, estimated_cycle_length)
    phase_probabilities = _phase_prior_from_cycle_day(cycle_day, estimated_cycle_length)

    return {
        "cycle_lengths": cycle_lengths,
        "prediction": {
            "last_period_start": last_period_start,
            "estimated_cycle_length": estimated_cycle_length,
            "predicted_start": predicted_start,
        },
        "phase_prior": {
            "cycle_day": cycle_day,
            "likely_phase": likely_phase,
            "phase_probabilities": phase_probabilities,
        },
    }
