from typing import Dict


PHASE_INDEX = {
    "Menstrual": 0,
    "Follicular": 1,
    "Fertility": 2,
    "Luteal": 3,
}


def phase_distance(a: str, b: str) -> int:
    return abs(PHASE_INDEX[a] - PHASE_INDEX[b])


def get_timing_status(layer1: Dict, layer2: Dict, bleeding_today: bool = False) -> str:
    layer1_phase = max(layer1["phase_probs"], key=layer1["phase_probs"].get)
    layer2_phase = layer2["top_phase"]

    if bleeding_today:
        return "Period started"

    if layer1_phase == layer2_phase:
        return "On track"

    dist = phase_distance(layer1_phase, layer2_phase)

    if dist == 1:
        if layer1_phase == "Follicular" and layer2_phase == "Fertility":
            return "Possibly earlier than expected"
        if layer1_phase == "Fertility" and layer2_phase == "Follicular":
            return "Possibly later than expected"
        if layer1_phase == "Fertility" and layer2_phase == "Luteal":
            return "Possibly earlier than expected"
        if layer1_phase == "Luteal" and layer2_phase == "Fertility":
            return "Possibly later than expected"
        if layer1_phase == "Luteal" and layer2_phase == "Menstrual":
            return "Period approaching"
        return "Timing uncertain this cycle"

    return "Timing uncertain this cycle"


def build_timing_note(layer1: Dict, layer2: Dict, timing_status: str, bleeding_today: bool = False) -> str:
    layer1_phase = max(layer1["phase_probs"], key=layer1["phase_probs"].get)
    layer2_phase = layer2["top_phase"]
    fertility_status = layer2["fertility_status"]

    if bleeding_today:
        return "Bleeding was logged today, so the cycle is treated as menstrual."

    if timing_status == "On track":
        return f"History and body signals both support a {layer2_phase.lower()} pattern today."

    if timing_status == "Possibly earlier than expected":
        return (
            f"History suggests {layer1_phase.lower()}, but body signals look a little further ahead "
            f"toward {layer2_phase.lower()}."
        )

    if timing_status == "Possibly later than expected":
        return (
            f"History suggests {layer1_phase.lower()}, but body signals look a little earlier "
            f"than expected."
        )

    if timing_status == "Period approaching":
        return (
            f"History suggests late luteal timing, and body signals may indicate that the period is approaching."
        )

    return (
        f"History suggests {layer1_phase.lower()}, while body signals lean {layer2_phase.lower()} "
        f"({fertility_status}). The pattern is not specific enough to confidently change the phase."
    )


def get_layer3_output(layer1: Dict, layer2: Dict, bleeding_today: bool = False) -> Dict:
    timing_status = get_timing_status(layer1, layer2, bleeding_today=bleeding_today)
    timing_note = build_timing_note(layer1, layer2, timing_status, bleeding_today=bleeding_today)

    return {
        "timing_status": timing_status,
        "timing_note": timing_note,
        "history_phase": max(layer1["phase_probs"], key=layer1["phase_probs"].get),
        "symptom_phase": layer2["top_phase"],
    }
