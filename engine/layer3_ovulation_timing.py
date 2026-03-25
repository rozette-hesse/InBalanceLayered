from typing import Dict


def get_timing_status(layer1: Dict, layer2: Dict) -> str:
    layer1_phase = max(layer1["phase_probs"], key=layer1["phase_probs"].get)
    layer2_phase = layer2["top_phase"]

    if layer1_phase == layer2_phase:
        return "On track"

    nearby_pairs = {
        ("Follicular", "Fertility"),
        ("Fertility", "Follicular"),
        ("Fertility", "Luteal"),
        ("Luteal", "Fertility"),
        ("Luteal", "Menstrual"),
        ("Menstrual", "Luteal"),
        ("Menstrual", "Follicular"),
        ("Follicular", "Menstrual"),
    }

    if (layer1_phase, layer2_phase) in nearby_pairs:
        if layer1_phase == "Luteal" and layer2_phase in {"Follicular", "Fertility"}:
            return "Possibly later than expected"
        if layer1_phase == "Follicular" and layer2_phase in {"Fertility", "Luteal"}:
            return "Possibly earlier than expected"
        if layer1_phase == "Fertility" and layer2_phase == "Luteal":
            return "Possibly earlier than expected"
        if layer1_phase == "Fertility" and layer2_phase == "Follicular":
            return "Possibly later than expected"
        if layer1_phase == "Luteal" and layer2_phase == "Menstrual":
            return "Possibly earlier than expected"
        if layer1_phase == "Menstrual" and layer2_phase == "Follicular":
            return "On track"

    return "Timing uncertain this cycle"


def build_timing_note(layer1: Dict, layer2: Dict, timing_status: str) -> str:
    layer1_phase = max(layer1["phase_probs"], key=layer1["phase_probs"].get)
    layer2_phase = layer2["top_phase"]
    fertility_status = layer2["fertility_status"]

    if timing_status == "On track":
        return f"History and body signals both support a {layer2_phase.lower()} pattern today."

    if timing_status == "Possibly earlier than expected":
        return (
            f"History suggests {layer1_phase.lower()}, but body signals look more {layer2_phase.lower()}. "
            f"This cycle may be moving a little earlier than usual."
        )

    if timing_status == "Possibly later than expected":
        return (
            f"History suggests {layer1_phase.lower()}, but body signals look more {layer2_phase.lower()}. "
            f"This cycle may be running a little later than expected."
        )

    return (
        f"History suggests {layer1_phase.lower()}, while body signals lean {layer2_phase.lower()} "
        f"({fertility_status}). This cycle looks less consistent than usual."
    )


def get_layer3_output(layer1: Dict, layer2: Dict) -> Dict:
    timing_status = get_timing_status(layer1, layer2)
    timing_note = build_timing_note(layer1, layer2, timing_status)

    return {
        "timing_status": timing_status,
        "timing_note": timing_note,
        "history_phase": max(layer1["phase_probs"], key=layer1["phase_probs"].get),
        "symptom_phase": layer2["top_phase"],
    }
