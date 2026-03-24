from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .config import PHASES
from .utils import parse_date, days_between, normalize_probs


def compute_cycle_lengths(period_starts: List[str]) -> List[int]:
    dates = sorted(parse_date(d) for d in period_starts)
    if len(dates) < 2:
        return []
    return [days_between(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]


def weighted_recent_cycle_length(lengths: List[int]) -> Optional[float]:
    if not lengths:
        return None
    weights = list(range(1, len(lengths) + 1))
    return sum(w * x for w, x in zip(weights, lengths)) / sum(weights)


def estimate_cycle_day(period_starts: List[str], today: Optional[str] = None) -> Optional[int]:
    if not period_starts:
        return None
    latest = max(parse_date(d) for d in period_starts)
    today_dt = parse_date(today) if today else datetime.today()
    return max((today_dt - latest).days + 1, 1)


def phase_probs_from_cycle_day(cycle_day: Optional[int], cycle_length: Optional[float]) -> Dict[str, float]:
    if cycle_day is None or cycle_length is None:
        return {p: 0.25 for p in PHASES}

    ovulation_day = round(cycle_length - 14)

    probs = {
        "Menstrual": 0.05,
        "Follicular": 0.10,
        "Fertility": 0.10,
        "Luteal": 0.10,
    }

    if 1 <= cycle_day <= 5:
        probs["Menstrual"] += 0.75
    elif 6 <= cycle_day <= max(ovulation_day - 3, 6):
        probs["Follicular"] += 0.70
    elif max(ovulation_day - 2, 1) <= cycle_day <= ovulation_day + 1:
        probs["Fertility"] += 0.75
    elif cycle_day > ovulation_day + 1:
        probs["Luteal"] += 0.75

    return normalize_probs(probs)


def get_layer1_output(period_starts: List[str], today: Optional[str] = None) -> Dict[str, object]:
    cycle_lengths = compute_cycle_lengths(period_starts)
    avg_cycle_length = weighted_recent_cycle_length(cycle_lengths)
    cycle_day = estimate_cycle_day(period_starts, today=today)

    if period_starts and avg_cycle_length is not None:
        latest = max(parse_date(d) for d in period_starts)
        predicted_next_period = (latest + timedelta(days=round(avg_cycle_length))).strftime("%Y-%m-%d")
    else:
        predicted_next_period = None

    return {
        "cycle_lengths": cycle_lengths,
        "estimated_cycle_length": avg_cycle_length,
        "cycle_day": cycle_day,
        "predicted_next_period": predicted_next_period,
        "phase_probs": phase_probs_from_cycle_day(cycle_day, avg_cycle_length),
    }
