from typing import Dict, List

from .config import PHASES, LAYER1_WEIGHT, LAYER2_WEIGHT
from .layer1_period_predictor import get_layer1_output
from .layer2_model_predictor import get_layer2_output
from .utils import normalize_probs


def has_symptom_input(symptoms: List[str], cervical_mucus: str, appetite: int = 0, exerciselevel: int = 0) -> bool:
    return (
        bool(symptoms)
        or ((cervical_mucus or "unknown").lower() != "unknown")
        or int(appetite) != 0
        or int(exerciselevel) != 0
    )


def fuse_phase_probs(layer1_probs: Dict[str, float], layer2_probs: Dict[str, float]) -> Dict[str, float]:
    fused = {
        phase: LAYER1_WEIGHT * layer1_probs.get(phase, 0.0) +
               LAYER2_WEIGHT * layer2_probs.get(phase, 0.0)
        for phase in PHASES
    }
    return normalize_probs(fused)


def get_fused_output(
    period_starts,
    symptoms=None,
    cervical_mucus="unknown",
    appetite: int = 0,
    exerciselevel: int = 0,
    today=None,
):
    symptoms = symptoms or []
    layer1 = get_layer1_output(period_starts, today=today)

    if not has_symptom_input(symptoms, cervical_mucus, appetite, exerciselevel):
        return {
            "mode": "layer1_only",
            "layer1": layer1,
            "layer2": None,
            "final_phase_probs": layer1["phase_probs"],
            "final_phase": max(layer1["phase_probs"], key=layer1["phase_probs"].get),
        }

    layer2 = get_layer2_output(
        symptoms=symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
    )
    fused_probs = fuse_phase_probs(layer1["phase_probs"], layer2["phase_probs"])

    return {
        "mode": "fused",
        "layer1": layer1,
        "layer2": layer2,
        "final_phase_probs": fused_probs,
        "final_phase": max(fused_probs, key=fused_probs.get),
    }
