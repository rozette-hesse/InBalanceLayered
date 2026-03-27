from typing import Dict, List, Optional

from .config import PHASES, LAYER1_WEIGHT, LAYER2_WEIGHT
from .layer1_period_predictor import get_layer1_output
from .layer2_model_predictor import get_layer2_output
from .layer3_ovulation_timing import get_layer3_output
from .utils import normalize_probs


PHASE_INDEX = {
    "Menstrual": 0,
    "Follicular": 1,
    "Fertility": 2,
    "Luteal": 3,
}


def has_symptom_input(
    symptoms: Optional[List[str]],
    cervical_mucus: str,
    appetite: int = 0,
    exerciselevel: int = 0,
    bleeding_today: bool = False,
) -> bool:
    return (
        bool(symptoms)
        or ((cervical_mucus or "unknown").lower() != "unknown")
        or int(appetite) != 0
        or int(exerciselevel) != 0
        or bool(bleeding_today)
    )


def phase_distance(a: str, b: str) -> int:
    return abs(PHASE_INDEX[a] - PHASE_INDEX[b])


def get_allowed_phases(
    baseline_phase: str,
    bleeding_today: bool = False,
) -> List[str]:
    if bleeding_today:
        return ["Menstrual"]

    if baseline_phase == "Menstrual":
        return ["Menstrual", "Follicular"]

    if baseline_phase == "Follicular":
        return ["Follicular", "Fertility"]

    if baseline_phase == "Fertility":
        return ["Follicular", "Fertility", "Luteal"]

    if baseline_phase == "Luteal":
        return ["Luteal", "Menstrual"]

    return PHASES


def constrain_probs_by_flow(
    fused_probs: Dict[str, float],
    baseline_phase: str,
    layer2: Dict[str, object],
    bleeding_today: bool = False,
) -> Dict[str, float]:
    allowed = set(get_allowed_phases(baseline_phase, bleeding_today=bleeding_today))

    top_prob = layer2.get("top_prob", 0.0)
    prob_gap = layer2.get("prob_gap", 0.0)
    signal_confidence = layer2.get("signal_confidence", "low")
    symptom_phase = layer2.get("top_phase", baseline_phase)

    # if signal is weak and contradictory, keep baseline tighter
    if (
        phase_distance(baseline_phase, symptom_phase) >= 1
        and signal_confidence == "low"
        and prob_gap < 0.20
        and not bleeding_today
    ):
        allowed = {baseline_phase}

    constrained = {}
    for phase, value in fused_probs.items():
        if phase in allowed:
            constrained[phase] = value
        else:
            constrained[phase] = 0.0

    total = sum(constrained.values())
    if total <= 0:
        constrained = {p: 1.0 if p == baseline_phase else 0.0 for p in PHASES}
        total = sum(constrained.values())

    constrained = {k: v / total for k, v in constrained.items()}

    # if baseline is luteal and menstrual wins without bleeding, prefer luteal + period approaching
    if (
        baseline_phase == "Luteal"
        and not bleeding_today
        and max(constrained, key=constrained.get) == "Menstrual"
    ):
        constrained["Luteal"] += 0.20
        total = sum(constrained.values())
        constrained = {k: v / total for k, v in constrained.items()}

    return constrained


def fuse_phase_probs(
    layer1_probs: Dict[str, float],
    layer2_probs: Dict[str, float],
) -> Dict[str, float]:
    fused = {
        phase: (
            LAYER1_WEIGHT * layer1_probs.get(phase, 0.0)
            + LAYER2_WEIGHT * layer2_probs.get(phase, 0.0)
        )
        for phase in PHASES
    }
    return normalize_probs(fused)


def get_fused_output(
    period_starts: List[str],
    symptoms: Optional[List[str]] = None,
    cervical_mucus: str = "unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    bleeding_today: bool = False,
    today: Optional[str] = None,
) -> Dict[str, object]:
    symptoms = symptoms or []

    layer1 = get_layer1_output(period_starts, today=today)
    baseline_phase = max(layer1["phase_probs"], key=layer1["phase_probs"].get)

    if not has_symptom_input(
        symptoms,
        cervical_mucus,
        appetite,
        exerciselevel,
        bleeding_today=bleeding_today,
    ):
        final_phase_probs = layer1["phase_probs"]
        final_phase = max(final_phase_probs, key=final_phase_probs.get)

        return {
            "mode": "layer1_only",
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
        bleeding_today=bleeding_today,
    )

    fused_probs = fuse_phase_probs(layer1["phase_probs"], layer2["phase_probs"])
    constrained_probs = constrain_probs_by_flow(
        fused_probs=fused_probs,
        baseline_phase=baseline_phase,
        layer2=layer2,
        bleeding_today=bleeding_today,
    )

    final_phase = max(constrained_probs, key=constrained_probs.get)

    layer3 = get_layer3_output(
        layer1=layer1,
        layer2={**layer2, "top_phase": final_phase},
        bleeding_today=bleeding_today,
    )

    return {
        "mode": "fused",
        "layer1": layer1,
        "layer2": layer2,
        "layer3": layer3,
        "final_phase_probs": constrained_probs,
        "final_phase": final_phase,
    }
