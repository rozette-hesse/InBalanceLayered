from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


PHASES = ["Fertility", "Follicular", "Luteal", "Menstrual"]


@dataclass
class LayerFusionConfig:
    # Main fusion weights
    layer1_weight: float = 0.20
    layer2_weight: float = 0.80

    # Phase-specific light calibration on Layer 2
    menstrual_boost_from_layer2: float = 1.10
    fertility_boost_from_layer2: float = 1.15
    luteal_boost_from_layer2: float = 1.00
    follicular_boost_from_layer2: float = 0.95

    # Confidence threshold
    low_confidence_threshold: float = 0.45


def normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    clean = {p: max(0.0, float(probs.get(p, 0.0))) for p in PHASES}
    total = sum(clean.values())
    if total <= 0:
        return {p: 1.0 / len(PHASES) for p in PHASES}
    return {p: v / total for p, v in clean.items()}


def confidence_from_probs(probs: Dict[str, float]) -> float:
    ordered = sorted(probs.values(), reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    return round(top1 - top2, 4)


def expected_phase_from_cycle_day(cycle_day: int, cycle_length: int = 28) -> str:
    """
    Simple Layer 1-style expected phase mapping.
    Replace later with your stronger Layer 1 output if needed.
    """
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


def layer1_phase_prior(cycle_day: int, cycle_length: int = 28) -> Dict[str, float]:
    """
    Converts Layer 1 cycle timing into soft phase probabilities.
    """
    phase = expected_phase_from_cycle_day(cycle_day, cycle_length)

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

    return normalize_probs(priors)


def apply_layer2_phase_adjustments(
    layer2_probs: Dict[str, float],
    config: LayerFusionConfig,
) -> Dict[str, float]:
    """
    Small calibration step on Layer 2 probabilities.
    """
    adjusted = dict(layer2_probs)

    adjusted["Menstrual"] *= config.menstrual_boost_from_layer2
    adjusted["Fertility"] *= config.fertility_boost_from_layer2
    adjusted["Luteal"] *= config.luteal_boost_from_layer2
    adjusted["Follicular"] *= config.follicular_boost_from_layer2

    return normalize_probs(adjusted)


def fuse_layer1_layer2(
    layer1_probs: Dict[str, float],
    layer2_probs: Dict[str, float],
    config: Optional[LayerFusionConfig] = None,
) -> Dict[str, object]:
    """
    Fuse Layer 1 and Layer 2 phase probabilities.
    """
    config = config or LayerFusionConfig()

    p1 = normalize_probs(layer1_probs)
    p2 = normalize_probs(layer2_probs)
    p2 = apply_layer2_phase_adjustments(p2, config)

    fused = {}
    for phase in PHASES:
        fused[phase] = (
            config.layer1_weight * p1[phase]
            + config.layer2_weight * p2[phase]
        )

    fused = normalize_probs(fused)
    predicted_phase = max(fused, key=fused.get)
    confidence = confidence_from_probs(fused)
    uncertainty_flag = confidence < config.low_confidence_threshold

    return {
        "predicted_phase": predicted_phase,
        "phase_probabilities": fused,
        "confidence": confidence,
        "uncertainty_flag": uncertainty_flag,
        "layer1_probs": p1,
        "layer2_probs": p2,
    }


def fuse_from_cycle_day_and_layer2(
    cycle_day: int,
    cycle_length: int,
    layer2_probs: Dict[str, float],
    config: Optional[LayerFusionConfig] = None,
) -> Dict[str, object]:
    """
    Convenience wrapper:
    - builds Layer 1 prior from cycle timing
    - fuses it with Layer 2 probabilities
    """
    layer1_probs = layer1_phase_prior(cycle_day=cycle_day, cycle_length=cycle_length)
    return fuse_layer1_layer2(layer1_probs, layer2_probs, config=config)


def estimate_period_shift_days(
    fused_phase_probs: Dict[str, float],
    cycle_day: int,
    cycle_length: int = 28,
) -> int:
    """
    First-pass rule-based period shift from fused phase probabilities.

    Negative = next period expected earlier
    Positive = next period expected later
    """
    probs = normalize_probs(fused_phase_probs)

    menstrual_p = probs["Menstrual"]
    fertility_p = probs["Fertility"]
    luteal_p = probs["Luteal"]

    # Strong menstrual evidence -> period may come earlier
    if menstrual_p >= 0.60:
        return -2
    if menstrual_p >= 0.45:
        return -1

    # Fertility still high late in cycle -> possible delayed ovulation
    if cycle_day >= 14 and fertility_p >= 0.45:
        return +3
    if cycle_day >= 16 and fertility_p >= 0.30:
        return +2

    # Strong luteal evidence -> keep baseline
    if luteal_p >= 0.50:
        return 0

    return 0


def predict_with_fusion(
    cycle_day: int,
    cycle_length: int,
    layer2_probs: Dict[str, float],
    config: Optional[LayerFusionConfig] = None,
) -> Dict[str, object]:
    """
    Main app-facing helper.
    """
    fused = fuse_from_cycle_day_and_layer2(
        cycle_day=cycle_day,
        cycle_length=cycle_length,
        layer2_probs=layer2_probs,
        config=config,
    )

    shift_days = estimate_period_shift_days(
        fused_phase_probs=fused["phase_probabilities"],
        cycle_day=cycle_day,
        cycle_length=cycle_length,
    )

    fused["period_shift_days"] = shift_days
    return fused


if __name__ == "__main__":
    example_layer2 = {
        "Menstrual": 0.10,
        "Follicular": 0.18,
        "Fertility": 0.12,
        "Luteal": 0.60,
    }

    result = predict_with_fusion(
        cycle_day=22,
        cycle_length=29,
        layer2_probs=example_layer2,
    )

    print(result)
