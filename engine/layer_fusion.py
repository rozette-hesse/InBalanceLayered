from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]


@dataclass
class LayerFusionConfig:
    layer1_weight: float = 0.20
    layer2_weight: float = 0.80

    menstrual_boost_from_layer2: float = 1.10
    fertility_boost_from_layer2: float = 1.15
    luteal_boost_from_layer2: float = 1.00
    follicular_boost_from_layer2: float = 0.95

    low_confidence_threshold: float = 0.45


def normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in probs.values())
    if total <= 0:
        return {p: 1.0 / len(PHASES) for p in PHASES}
    return {p: max(v, 0.0) / total for p, v in probs.items()}


def confidence_from_probs(probs: Dict[str, float]) -> float:
    vals = sorted(probs.values(), reverse=True)
    top1 = vals[0]
    top2 = vals[1] if len(vals) > 1 else 0.0
    return round(top1 - top2, 4)


def apply_layer2_phase_adjustments(
    layer2_probs: Dict[str, float],
    config: LayerFusionConfig,
) -> Dict[str, float]:
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

    return {
        "predicted_phase": predicted_phase,
        "phase_probabilities": fused,
        "confidence": confidence,
        "uncertainty_flag": confidence < config.low_confidence_threshold,
        "layer1_probs": p1,
        "layer2_probs": p2,
    }


def estimate_period_shift_days(
    fused_phase_probs: Dict[str, float],
    symptoms: list[str],
    cervical_mucus: str,
    cycle_day: int,
    cycle_length: int = 28,
) -> int:
    """
    Stable shift logic:
    - menstrual-like evidence can move date earlier
    - fertility-like mucus can move date later
    - symptoms matter, but mucus remains strongest
    """
    probs = normalize_probs(fused_phase_probs)
    symptoms = symptoms or []

    menstrual_p = probs["Menstrual"]
    fertility_p = probs["Fertility"]
    luteal_p = probs["Luteal"]

    has = lambda s: s in symptoms

    shift = 0

    # Early shift if menstrual evidence is meaningful
    if menstrual_p >= 0.55:
        shift -= 2
    elif menstrual_p >= 0.40:
        shift -= 1

    if has("cramps") and has("bloating"):
        shift -= 1

    # Later shift if fertility evidence is still strong
    if cervical_mucus in ["watery", "eggwhite"]:
        if cycle_day >= 12:
            shift += 2
        else:
            shift += 1

    if fertility_p >= 0.45 and cycle_day >= 14:
        shift += 1

    # If strong luteal alignment, don't move much
    if luteal_p >= 0.50 and shift == 0:
        shift = 0

    return max(-3, min(5, shift))


def predict_with_fusion(
    layer1_phase_probs: Dict[str, float],
    layer2_probs: Dict[str, float],
    symptoms: list[str],
    cervical_mucus: str,
    cycle_day: int,
    cycle_length: int,
    config: Optional[LayerFusionConfig] = None,
) -> Dict[str, object]:
    fused = fuse_layer1_layer2(
        layer1_probs=layer1_phase_probs,
        layer2_probs=layer2_probs,
        config=config,
    )

    shift_days = estimate_period_shift_days(
        fused_phase_probs=fused["phase_probabilities"],
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        cycle_day=cycle_day,
        cycle_length=cycle_length,
    )

    fused["period_shift_days"] = shift_days
    return fused
