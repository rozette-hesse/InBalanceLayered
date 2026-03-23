import streamlit as st
import pandas as pd

# =========================================================
# SIMPLE INBALANCE STREAMLIT GUI
# Layer 1 + Layer 2A + Fusion
# =========================================================

PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

SYMPTOMS = [
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
]

MUCUS_OPTIONS = ["unknown", "dry", "sticky", "creamy", "watery", "eggwhite"]


# =========================================================
# LAYER 1
# =========================================================
def normalize_probs(probs: dict) -> dict:
    total = sum(max(v, 0.0) for v in probs.values())
    if total == 0:
        return {k: 1 / len(probs) for k in probs}
    return {k: max(v, 0.0) / total for k, v in probs.items()}


def expected_phase_from_cycle_day(cycle_day: int, cycle_length: int = 28) -> str:
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


def layer1_phase_prior(cycle_day: int, cycle_length: int = 28) -> dict:
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


# =========================================================
# LAYER 2A
# SIMPLE APPROXIMATION FOR GUI DEMO
# Replace later with your trained model if needed
# =========================================================
def layer2a_symptom_mucus_engine(symptoms: dict, mucus: str) -> dict:
    score = {
        "Menstrual": 0.05,
        "Follicular": 0.08,
        "Fertility": 0.08,
        "Luteal": 0.08,
    }

    # symptom logic
    if symptoms.get("cramps", False):
        score["Menstrual"] += 0.28

    if symptoms.get("bloating", False):
        score["Menstrual"] += 0.08
        score["Luteal"] += 0.08

    if symptoms.get("sorebreasts", False):
        score["Luteal"] += 0.20

    if symptoms.get("foodcravings", False):
        score["Luteal"] += 0.12

    if symptoms.get("moodswing", False):
        score["Luteal"] += 0.10

    if symptoms.get("sleepissue", False):
        score["Luteal"] += 0.08

    if symptoms.get("fatigue", False):
        score["Luteal"] += 0.08
        score["Menstrual"] += 0.05

    if symptoms.get("headaches", False):
        score["Menstrual"] += 0.05
        score["Luteal"] += 0.05

    if symptoms.get("stress", False):
        score["Luteal"] += 0.05

    if symptoms.get("indigestion", False):
        score["Luteal"] += 0.04

    # mucus logic
    if mucus == "dry":
        score["Follicular"] += 0.06
    elif mucus == "sticky":
        score["Follicular"] += 0.08
    elif mucus == "creamy":
        score["Follicular"] += 0.16
        score["Fertility"] += 0.08
    elif mucus == "watery":
        score["Fertility"] += 0.35
    elif mucus == "eggwhite":
        score["Fertility"] += 0.48

    if mucus in ["watery", "eggwhite"]:
        score["Menstrual"] = max(0.01, score["Menstrual"] - 0.08)

    return normalize_probs(score)


# =========================================================
# FUSION
# =========================================================
def apply_layer2_phase_adjustments(
    layer2_probs: dict,
    menstrual_boost: float = 1.10,
    fertility_boost: float = 1.15,
    luteal_boost: float = 1.00,
    follicular_boost: float = 0.95,
) -> dict:
    adjusted = dict(layer2_probs)
    adjusted["Menstrual"] *= menstrual_boost
    adjusted["Fertility"] *= fertility_boost
    adjusted["Luteal"] *= luteal_boost
    adjusted["Follicular"] *= follicular_boost
    return normalize_probs(adjusted)


def fuse_layer1_layer2(
    layer1_probs: dict,
    layer2_probs: dict,
    layer1_weight: float = 0.20,
    layer2_weight: float = 0.80,
) -> dict:
    p1 = normalize_probs(layer1_probs)
    p2 = normalize_probs(layer2_probs)
    p2 = apply_layer2_phase_adjustments(p2)

    fused = {}
    for phase in PHASES:
        fused[phase] = layer1_weight * p1[phase] + layer2_weight * p2[phase]

    return normalize_probs(fused)


def confidence_from_probs(probs: dict) -> float:
    ordered = sorted(probs.values(), reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    return top1 - top2


def predict_with_fusion(
    cycle_day: int,
    cycle_length: int,
    symptoms: dict,
    mucus: str,
    layer1_weight: float,
    layer2_weight: float,
):
    layer1_probs = layer1_phase_prior(cycle_day, cycle_length)
    layer2_probs = layer2a_symptom_mucus_engine(symptoms, mucus)
    fused_probs = fuse_layer1_layer2(
        layer1_probs=layer1_probs,
        layer2_probs=layer2_probs,
        layer1_weight=layer1_weight,
        layer2_weight=layer2_weight,
    )

    predicted_phase = max(fused_probs, key=fused_probs.get)
    confidence = confidence_from_probs(fused_probs)

    return {
        "predicted_phase": predicted_phase,
        "confidence": confidence,
        "layer1_probs": layer1_probs,
        "layer2_probs": layer2_probs,
        "fused_probs": fused_probs,
    }


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="InBalance Cycle Engine", layout="wide")

st.title("InBalance Cycle Engine")
st.caption("Layer 1 + Layer 2A + Fusion demo")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Cycle inputs")
    cycle_day = st.number_input("Cycle day", min_value=1, max_value=60, value=22, step=1)
    cycle_length = st.number_input("Cycle length", min_value=20, max_value=45, value=28, step=1)
    mucus = st.selectbox("Cervical mucus", MUCUS_OPTIONS, index=5)

    st.subheader("Symptoms")
    symptoms = {}
    symptom_cols = st.columns(2)
    for i, symptom in enumerate(SYMPTOMS):
        with symptom_cols[i % 2]:
            symptoms[symptom] = st.checkbox(symptom, value=symptom in ["sorebreasts", "fatigue", "foodcravings", "bloating"])

with col2:
    st.subheader("Fusion weights")
    layer1_weight = st.slider("Layer 1 weight", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
    layer2_weight = round(1.0 - layer1_weight, 2)
    st.write(f"Layer 2 weight: **{layer2_weight}**")

    result = predict_with_fusion(
        cycle_day=int(cycle_day),
        cycle_length=int(cycle_length),
        symptoms=symptoms,
        mucus=mucus,
        layer1_weight=layer1_weight,
        layer2_weight=layer2_weight,
    )

    st.subheader("Final output")
    st.metric("Predicted phase", result["predicted_phase"])
    st.metric("Confidence", f"{result['confidence']:.2f}")

    st.write("Expected phase from Layer 1:", expected_phase_from_cycle_day(int(cycle_day), int(cycle_length)))

st.divider()

tab1, tab2, tab3 = st.tabs(["Layer 1", "Layer 2A", "Fusion"])

with tab1:
    st.subheader("Layer 1 probabilities")
    layer1_df = pd.DataFrame(
        {"Phase": list(result["layer1_probs"].keys()), "Probability": list(result["layer1_probs"].values())}
    )
    st.dataframe(layer1_df, use_container_width=True)
    for phase, prob in result["layer1_probs"].items():
        st.write(f"**{phase}**")
        st.progress(float(prob))

with tab2:
    st.subheader("Layer 2A probabilities")
    layer2_df = pd.DataFrame(
        {"Phase": list(result["layer2_probs"].keys()), "Probability": list(result["layer2_probs"].values())}
    )
    st.dataframe(layer2_df, use_container_width=True)
    for phase, prob in result["layer2_probs"].items():
        st.write(f"**{phase}**")
        st.progress(float(prob))

with tab3:
    st.subheader("Fused probabilities")
    fused_df = pd.DataFrame(
        {"Phase": list(result["fused_probs"].keys()), "Probability": list(result["fused_probs"].values())}
    ).sort_values("Probability", ascending=False)
    st.dataframe(fused_df, use_container_width=True)
    for phase, prob in result["fused_probs"].items():
        st.write(f"**{phase}**")
        st.progress(float(prob))
