from typing import Dict, List, Optional

from .config import LAYER1_WEIGHT, LAYER2_WEIGHT, NON_MENSTRUAL_PHASES, PHASES
from .layer1_period_predictor import get_layer1_output
from .layer2_model_predictor import get_layer2_output
from .layer3_ovulation_timing import get_layer3_output
from .utils import normalize_probs


def has_symptom_input(
    symptoms: Optional[List[str]],
    cervical_mucus: str,
    appetite: int = 0,
    exerciselevel: int = 0,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
) -> bool:
    return (
        bool(symptoms)
        or ((cervical_mucus or "unknown").lower() != "unknown")
        or int(appetite) != 0
        or int(exerciselevel) != 0
        or bool(recent_daily_logs)
    )


def _map_layer1_to_non_menstrual(layer1_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Layer 1 still has 4 classes.
    For fusion before a period is logged:
      - fold Menstrual mass into Luteal
      - keep Follicular / Fertility / Luteal
    """
    mapped = {
        "Follicular": float(layer1_probs.get("Follicular", 0.0)),
        "Fertility": float(layer1_probs.get("Fertility", 0.0)),
        "Luteal": float(layer1_probs.get("Luteal", 0.0) + layer1_probs.get("Menstrual", 0.0)),
    }
    return normalize_probs(mapped)


def _get_layer1_non_menstrual_top_phase(layer1_probs_3: Dict[str, float]) -> str:
    return max(layer1_probs_3, key=layer1_probs_3.get)


def _get_allowed_phases(baseline_phase: str) -> List[str]:
    if baseline_phase == "Follicular":
        return ["Follicular", "Fertility"]
    if baseline_phase == "Fertility":
        return ["Follicular", "Fertility", "Luteal"]
    if baseline_phase == "Luteal":
        return ["Follicular", "Fertility", "Luteal"]
    return NON_MENSTRUAL_PHASES


def _constrain_non_menstrual_probs(
    fused_probs: Dict[str, float],
    baseline_phase: str,
    layer2: Dict[str, object],
) -> Dict[str, float]:
    allowed = set(_get_allowed_phases(baseline_phase))

    top_prob = layer2.get("top_prob", 0.0)
    prob_gap = layer2.get("prob_gap", 0.0)
    signal_confidence = layer2.get("signal_confidence", "low")
    symptom_phase = layer2.get("top_phase", baseline_phase)

    # If layer2 is weak and contradictory, keep baseline tighter
    if (
        signal_confidence == "low"
        and prob_gap < 0.20
        and baseline_phase != symptom_phase
    ):
        allowed = {baseline_phase}

    constrained = {}
    for phase, value in fused_probs.items():
        constrained[phase] = value if phase in allowed else 0.0

    total = sum(constrained.values())
    if total <= 0:
        constrained = {p: 1.0 if p == baseline_phase else 0.0 for p in NON_MENSTRUAL_PHASES}
    else:
        constrained = {k: v / total for k, v in constrained.items()}

    return constrained


def _fuse_non_menstrual_probs(
    layer1_probs_3: Dict[str, float],
    layer2_probs_3: Dict[str, float],
) -> Dict[str, float]:
    fused = {
        phase: (
            LAYER1_WEIGHT * layer1_probs_3.get(phase, 0.0)
            + LAYER2_WEIGHT * layer2_probs_3.get(phase, 0.0)
        )
        for phase in NON_MENSTRUAL_PHASES
    }
    return normalize_probs(fused)


def get_fused_output(
    period_starts: List[str],
    symptoms: Optional[List[str]] = None,
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    period_start_logged: bool = False,
    recent_daily_logs: Optional[List[Dict[str, object]]] = None,
    today: Optional[str] = None,
) -> Dict[str, object]:
    symptoms = symptoms or []

    layer1 = get_layer1_output(period_starts, today=today)
    layer1_probs_3 = _map_layer1_to_non_menstrual(layer1["phase_probs"])
    layer1_non_menstrual_top = _get_layer1_non_menstrual_top_phase(layer1_probs_3)

    # Hard override for actual period start
    if period_start_logged:
        final_phase_probs = {
            "Menstrual": 1.0,
            "Follicular": 0.0,
            "Fertility": 0.0,
            "Luteal": 0.0,
        }

        return {
            "mode": "period_start_override",
            "layer1": layer1,
            "layer2": None,
            "layer3": {
                "timing_status": "Period started",
                "timing_note": "You logged a period start today, so the cycle resets and today is marked as menstrual.",
                "history_phase": layer1_non_menstrual_top,
                "symptom_phase": None,
            },
            "final_phase_probs": final_phase_probs,
            "final_phase": "Menstrual",
        }

    # No symptom input -> rely on timing only, but still keep result non-menstrual
    if not has_symptom_input(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
        recent_daily_logs=recent_daily_logs,
    ):
        final_non_menstrual = layer1_probs_3
        final_phase = max(final_non_menstrual, key=final_non_menstrual.get)

        final_phase_probs = {
            "Menstrual": 0.0,
            "Follicular": final_non_menstrual.get("Follicular", 0.0),
            "Fertility": final_non_menstrual.get("Fertility", 0.0),
            "Luteal": final_non_menstrual.get("Luteal", 0.0),
        }

        return {
            "mode": "layer1_only_non_menstrual",
            "layer1": layer1,
            "layer2": None,
            "layer3": None,
            "final_phase_probs": final_phase_probs,
            "final_phase": final_phase,
        }

    layer2 = get_layer2_output(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
        recent_daily_logs=recent_daily_logs,
    )

    fused_probs_3 = _fuse_non_menstrual_probs(layer1_probs_3, layer2["phase_probs"])
    constrained_probs_3 = _constrain_non_menstrual_probs(
        fused_probs=fused_probs_3,
        baseline_phase=layer1_non_menstrual_top,
        layer2=layer2,
    )

    final_phase = max(constrained_probs_3, key=constrained_probs_3.get)

    layer3 = get_layer3_output(
        layer1={**layer1, "top_phase": layer1_non_menstrual_top, "phase_probs": layer1_probs_3},
        layer2={**layer2, "top_phase": final_phase},
        period_start_logged=False,
    )

    final_phase_probs = {
        "Menstrual": 0.0,
        "Follicular": constrained_probs_3.get("Follicular", 0.0),
        "Fertility": constrained_probs_3.get("Fertility", 0.0),
        "Luteal": constrained_probs_3.get("Luteal", 0.0),
    }

    return {
        "mode": "fused_non_menstrual",
        "layer1": {**layer1, "non_menstrual_phase_probs": layer1_probs_3, "top_phase": layer1_non_menstrual_top},
        "layer2": layer2,
        "layer3": layer3,
        "final_phase_probs": final_phase_probs,
        "final_phase": final_phase,
    }
