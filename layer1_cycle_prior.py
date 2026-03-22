from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


DateLike = Union[str, pd.Timestamp]


# =========================================================
# Core utilities
# =========================================================

def _to_sorted_timestamps(period_starts: Sequence[DateLike]) -> List[pd.Timestamp]:
    """
    Convert input dates to sorted pandas Timestamps.
    Invalid/null dates are dropped.
    """
    timestamps = pd.to_datetime(list(period_starts), errors="coerce")
    timestamps = [ts.normalize() for ts in timestamps if pd.notnull(ts)]
    return sorted(timestamps)


def calculate_cycle_lengths(period_starts: Sequence[DateLike]) -> np.ndarray:
    """
    Calculate cycle lengths in days from consecutive period start dates.

    Example:
        starts = [2025-01-01, 2025-01-29, 2025-02-27]
        returns [28, 29]
    """
    starts = _to_sorted_timestamps(period_starts)

    if len(starts) < 2:
        return np.array([], dtype=float)

    return np.array(
        [(starts[i] - starts[i - 1]).days for i in range(1, len(starts))],
        dtype=float
    )


def classify_regularity(cycle_lengths: Sequence[float]) -> str:
    """
    Classify cycle pattern as:
    - unknown
    - regular
    - moderately_variable
    - irregular
    """
    values = np.array(cycle_lengths, dtype=float)

    if len(values) < 2:
        return "unknown"

    std = float(np.std(values))
    mean = float(np.mean(values))
    cv = std / mean if mean > 0 else 0.0

    if std <= 2 and cv <= 0.08:
        return "regular"
    if std <= 5 and cv <= 0.15:
        return "moderately_variable"
    return "irregular"


def estimate_variability(cycle_lengths: Sequence[float]) -> Dict[str, Optional[float]]:
    """
    Return variability statistics for cycle lengths.
    """
    values = np.array(cycle_lengths, dtype=float)

    if len(values) == 0:
        return {
            "std": None,
            "mad": None,
            "min": None,
            "max": None,
            "range": None,
        }

    if len(values) == 1:
        only = float(values[0])
        return {
            "std": 0.0,
            "mad": 0.0,
            "min": only,
            "max": only,
            "range": 0.0,
        }

    std = float(np.std(values))
    mad = float(np.median(np.abs(values - np.median(values))))

    return {
        "std": std,
        "mad": mad,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


# =========================================================
# Benchmark estimators
# =========================================================

def estimate_cycle_length_last(cycle_lengths: Sequence[float]) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)
    if len(values) == 0:
        return None
    return int(round(values[-1]))


def estimate_cycle_length_mean(cycle_lengths: Sequence[float]) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)
    if len(values) == 0:
        return None
    return int(round(np.mean(values)))


def estimate_cycle_length_median(cycle_lengths: Sequence[float]) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)
    if len(values) == 0:
        return None
    return int(round(np.median(values)))


def estimate_cycle_length_recent_mean(
    cycle_lengths: Sequence[float],
    recent_n: int = 3,
) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)
    if len(values) == 0:
        return None

    n = min(recent_n, len(values))
    return int(round(np.mean(values[-n:])))


def estimate_cycle_length_blended(cycle_lengths: Sequence[float]) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)
    if len(values) == 0:
        return None

    full_median = np.median(values)
    full_mean = np.mean(values)
    recent_n = min(3, len(values))
    recent_mean = np.mean(values[-recent_n:])

    estimate = 0.5 * recent_mean + 0.3 * full_median + 0.2 * full_mean
    return int(round(estimate))


def estimate_cycle_length_weighted_recent(cycle_lengths: Sequence[float]) -> Optional[int]:
    """
    Weighted average where recent cycles have larger weights.
    """
    values = np.array(cycle_lengths, dtype=float)

    if len(values) == 0:
        return None
    if len(values) == 1:
        return int(round(values[0]))

    weights = np.arange(1, len(values) + 1, dtype=float)
    weighted_mean = np.average(values, weights=weights)
    return int(round(weighted_mean))


def estimate_cycle_length_trimmed_mean(cycle_lengths: Sequence[float]) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)

    if len(values) == 0:
        return None
    if len(values) < 4:
        return int(round(np.mean(values)))

    sorted_vals = np.sort(values)
    trimmed = sorted_vals[1:-1]
    return int(round(np.mean(trimmed)))


def estimate_cycle_length_clipped_mean(cycle_lengths: Sequence[float]) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)

    if len(values) == 0:
        return None
    if len(values) < 3:
        return int(round(np.mean(values)))

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    clipped = np.clip(values, lower, upper)
    return int(round(np.mean(clipped)))


def estimate_cycle_length_ensemble(cycle_lengths: Sequence[float]) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)
    if len(values) == 0:
        return None

    preds = [
        estimate_cycle_length_mean(values),
        estimate_cycle_length_median(values),
        estimate_cycle_length_weighted_recent(values),
    ]
    preds = [p for p in preds if p is not None]

    if not preds:
        return None

    return int(round(np.mean(preds)))


def estimate_cycle_length_adaptive(cycle_lengths: Sequence[float]) -> Optional[int]:
    values = np.array(cycle_lengths, dtype=float)
    if len(values) == 0:
        return None

    regularity = classify_regularity(values)

    full_median = np.median(values)
    full_mean = np.mean(values)
    recent_n = min(3, len(values))
    recent_mean = np.mean(values[-recent_n:])

    if regularity == "regular":
        estimate = full_mean
    elif regularity == "moderately_variable":
        estimate = 0.5 * recent_mean + 0.3 * full_median + 0.2 * full_mean
    else:
        estimate = 0.6 * full_mean + 0.4 * full_median

    return int(round(estimate))


# =========================================================
# Final production estimator
# =========================================================

def estimate_cycle_length_production(cycle_lengths: Sequence[float]) -> Optional[int]:
    """
    Final recommended estimator for app use.

    Rules:
    - regular -> mean
    - moderately variable -> weighted recent
    - irregular -> weighted recent + median compromise

    Then lightly clip prediction to observed range ± 2 days.
    """
    values = np.array(cycle_lengths, dtype=float)

    if len(values) == 0:
        return None
    if len(values) == 1:
        return int(round(values[0]))

    regularity = classify_regularity(values)

    if regularity == "regular":
        pred = estimate_cycle_length_mean(values)
    elif regularity == "moderately_variable":
        pred = estimate_cycle_length_weighted_recent(values)
    else:
        pred = int(round(
            0.5 * estimate_cycle_length_weighted_recent(values)
            + 0.5 * estimate_cycle_length_median(values)
        ))

    observed_min = np.min(values)
    observed_max = np.max(values)

    lower = observed_min - 2
    upper = observed_max + 2

    pred = max(lower, min(pred, upper))
    return int(round(pred))


# =========================================================
# Confidence and uncertainty
# =========================================================

def get_confidence_label(cycle_lengths: Sequence[float], regularity: str) -> str:
    """
    Confidence depends on both history length and regularity.
    """
    n = len(cycle_lengths)

    if n < 2:
        return "Low"
    if n < 4:
        return "Moderate"

    if regularity == "regular":
        return "High"
    if regularity == "moderately_variable":
        return "Moderate"
    return "Low"


def get_prediction_window_days(cycle_lengths: Sequence[float], regularity: str) -> int:
    """
    Wider uncertainty windows for more variable cycles.
    """
    values = np.array(cycle_lengths, dtype=float)

    if len(values) < 2:
        return 3

    std = float(np.std(values))

    if regularity == "regular":
        return max(2, int(round(std + 1)))
    if regularity == "moderately_variable":
        return max(3, int(round(std + 2)))
    return max(5, int(round(std + 3)))


# =========================================================
# Main Layer 1 outputs
# =========================================================

def predict_next_period(period_starts: Sequence[DateLike]) -> Optional[Dict[str, Any]]:
    """
    Predict the next expected period using cycle history only.
    """
    starts = _to_sorted_timestamps(period_starts)

    if len(starts) < 2:
        return None

    cycle_lengths = calculate_cycle_lengths(starts)
    regularity = classify_regularity(cycle_lengths)
    variability = estimate_variability(cycle_lengths)
    estimated_cycle_length = estimate_cycle_length_production(cycle_lengths)
    confidence = get_confidence_label(cycle_lengths, regularity)
    window_days = get_prediction_window_days(cycle_lengths, regularity)

    if estimated_cycle_length is None:
        return None

    last_start = starts[-1]
    predicted_start = last_start + timedelta(days=estimated_cycle_length)

    return {
        "predicted_start": predicted_start,
        "prediction_start": predicted_start - timedelta(days=window_days),
        "prediction_end": predicted_start + timedelta(days=window_days),
        "estimated_cycle_length": estimated_cycle_length,
        "regularity": regularity,
        "variability": variability,
        "confidence": confidence,
        "window_days": window_days,
    }


def get_cycle_day(period_starts: Sequence[DateLike], current_date: Optional[DateLike] = None) -> Optional[int]:
    """
    Return current cycle day based on latest logged period start.
    Cycle day 1 = first day of last period.
    """
    starts = _to_sorted_timestamps(period_starts)

    if len(starts) < 1:
        return None

    if current_date is None:
        now = pd.Timestamp.today().normalize()
    else:
        now = pd.to_datetime(current_date, errors="coerce")

    if pd.isnull(now):
        return None

    last_start = starts[-1]
    cycle_day = (now.normalize() - last_start).days + 1

    if cycle_day < 1:
        return None

    return int(cycle_day)


def estimate_phase_prior(
    period_starts: Sequence[DateLike],
    current_date: Optional[DateLike] = None,
) -> Optional[Dict[str, Any]]:
    """
    Estimate cycle phase probabilities using cycle math only.
    This is a prior for Layer 2, not a final truth label.
    """
    starts = _to_sorted_timestamps(period_starts)

    if len(starts) < 2:
        return None

    if current_date is None:
        now = pd.Timestamp.today().normalize()
    else:
        now = pd.to_datetime(current_date, errors="coerce")

    if pd.isnull(now):
        return None

    cycle_lengths = calculate_cycle_lengths(starts)
    est_len = estimate_cycle_length_production(cycle_lengths)
    regularity = classify_regularity(cycle_lengths)
    cycle_day = get_cycle_day(starts, now)

    if est_len is None or cycle_day is None:
        return None

    # Approximate ovulation timing
    ovulation_day = est_len - 14 + 1

    # Widen uncertainty around ovulation for irregular cycles
    spread = 1 if regularity == "regular" else 2 if regularity == "moderately_variable" else 4

    probs = {
        "Menstrual": 0.0,
        "Follicular": 0.0,
        "Fertile": 0.0,
        "Luteal": 0.0,
    }

    if cycle_day <= 5:
        probs["Menstrual"] = 0.75
        probs["Follicular"] = 0.20
        probs["Luteal"] = 0.05
    elif cycle_day < ovulation_day - spread:
        probs["Follicular"] = 0.70
        probs["Fertile"] = 0.20
        probs["Luteal"] = 0.10
    elif ovulation_day - spread <= cycle_day <= ovulation_day + spread:
        probs["Fertile"] = 0.60
        probs["Follicular"] = 0.20
        probs["Luteal"] = 0.20
    else:
        probs["Luteal"] = 0.75
        probs["Fertile"] = 0.10
        probs["Follicular"] = 0.15

    likely_phase = max(probs, key=probs.get)

    return {
        "cycle_day": cycle_day,
        "estimated_cycle_length": est_len,
        "regularity": regularity,
        "likely_phase": likely_phase,
        "phase_probabilities": probs,
    }


def get_layer1_output(
    period_starts: Sequence[DateLike],
    current_date: Optional[DateLike] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper that returns all Layer 1 outputs together.
    """
    starts = _to_sorted_timestamps(period_starts)

    prediction = predict_next_period(starts)
    phase_prior = estimate_phase_prior(starts, current_date=current_date)
    cycle_lengths = calculate_cycle_lengths(starts)

    return {
        "n_logged_periods": len(starts),
        "cycle_lengths": cycle_lengths.tolist(),
        "prediction": prediction,
        "phase_prior": phase_prior,
    }
