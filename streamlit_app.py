import streamlit as st
import pandas as pd
from datetime import date, timedelta

from engine.layer1_period_predictor import get_layer1_output
from engine.layer2a_phase_predictor import get_layer2_output
from engine.layer_fusion import LayerFusionConfig, predict_with_fusion


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


st.set_page_config(page_title="InBalance Cycle Engine", layout="wide")

st.title("InBalance Cycle Engine")
st.caption("Simple and stable Layer 1 + Layer 2 + Fusion demo")


left, right = st.columns([1.2, 1])

with left:
    st.subheader("1) Period history")

    n_periods = st.number_input(
        "How many periods do you want to enter?",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
    )

    today_default = date.today()
    period_starts = []
    period_ends = []

    for i in range(int(n_periods)):
        st.markdown(f"**Period {i+1}**")
        c1, c2 = st.columns(2)

        with c1:
            start = st.date_input(
                f"Start date #{i+1}",
                value=today_default - timedelta(days=(int(n_periods) - i) * 30),
                key=f"start_{i}",
            )

        with c2:
            end = st.date_input(
                f"End date #{i+1}",
                value=today_default - timedelta(days=(int(n_periods) - i) * 30 - 4),
                key=f"end_{i}",
            )

        period_starts.append(start)
        period_ends.append(end)

    st.subheader("2) Symptom log")
    log_date = st.date_input("Symptom log date", value=today_default)
    cervical_mucus = st.selectbox("Cervical mucus", MUCUS_OPTIONS, index=0)
    symptoms_selected = st.multiselect(
        "Choose symptoms for that day",
        SYMPTOMS,
        default=[],
    )

with right:
    st.subheader("3) Fusion settings")
    layer1_weight = st.slider("Layer 1 weight", 0.0, 1.0, 0.20, 0.05)
    layer2_weight = round(1.0 - layer1_weight, 2)
    st.write(f"Layer 2 weight: **{layer2_weight}**")

    run_btn = st.button("Run prediction", type="primary", use_container_width=True)


if run_btn:
    errors = []
    for i, (s, e) in enumerate(zip(period_starts, period_ends), start=1):
        if e < s:
            errors.append(f"Period {i}: end date is before start date.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    period_starts_sorted = sorted(period_starts)

    # -------- Layer 1 --------
    try:
        layer1_out = get_layer1_output(
            period_starts=period_starts_sorted,
            current_date=log_date,
        )
    except Exception as e:
        st.error(f"Layer 1 failed: {e}")
        st.stop()

    prediction = layer1_out.get("prediction")
    phase_prior = layer1_out.get("phase_prior")

    if prediction is None or phase_prior is None:
        st.error("Layer 1 could not generate output.")
        st.stop()

    baseline_next_period = prediction["predicted_start"]
    estimated_cycle_length = prediction["estimated_cycle_length"]
    cycle_lengths = layer1_out["cycle_lengths"]
    cycle_day = phase_prior["cycle_day"]
    last_period_start = prediction["last_period_start"]

    # -------- Layer 2 --------
    try:
        layer2_out = get_layer2_output(
            log_date=log_date,
            last_period_start=last_period_start,
            cycle_day=cycle_day,
            cycle_length=estimated_cycle_length,
            symptoms=symptoms_selected,
            cervical_mucus=cervical_mucus,
        )
    except Exception as e:
        st.error(f"Layer 2 failed: {e}")
        st.stop()

    # -------- Fusion --------
    try:
        fusion_config = LayerFusionConfig(
            layer1_weight=layer1_weight,
            layer2_weight=layer2_weight,
            menstrual_boost_from_layer2=1.10,
            fertility_boost_from_layer2=1.15,
            luteal_boost_from_layer2=1.00,
            follicular_boost_from_layer2=0.95,
            low_confidence_threshold=0.45,
        )

        fusion_out = predict_with_fusion(
            layer1_phase_probs=phase_prior["phase_probabilities"],
            layer2_probs=layer2_out["phase_probabilities"],
            symptoms=symptoms_selected,
            cervical_mucus=cervical_mucus,
            cycle_day=cycle_day,
            cycle_length=estimated_cycle_length,
            config=fusion_config,
        )
    except Exception as e:
        st.error(f"Fusion failed: {e}")
        st.stop()

    shift_days = fusion_out["period_shift_days"]
    final_next_period = baseline_next_period + timedelta(days=shift_days)

    st.markdown("---")
    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Layer 1 baseline next period", str(baseline_next_period))
    c2.metric("Layer 2 predicted phase", str(layer2_out["predicted_phase"]))
    c3.metric("Fusion predicted phase", str(fusion_out["predicted_phase"]))

    c4, c5, c6 = st.columns(3)
    c4.metric("Estimated cycle length", f"{estimated_cycle_length} days")
    c5.metric("Cycle day on log date", cycle_day)
    c6.metric("Fusion shift", f"{shift_days:+d} day(s)")

    st.metric("Final adjusted next period", str(final_next_period))

    st.markdown("---")
    st.subheader("Layer comparison")

    a, b, c = st.columns(3)

    with a:
        st.markdown("**Layer 1**")
        st.write(f"Cycle lengths used: {cycle_lengths}")
        st.write(f"Likely phase: {phase_prior['likely_phase']}")
        st.dataframe(
            pd.DataFrame({
                "Phase": list(phase_prior["phase_probabilities"].keys()),
                "Probability": list(phase_prior["phase_probabilities"].values()),
            }).sort_values("Probability", ascending=False),
            use_container_width=True,
        )

    with b:
        st.markdown("**Layer 2**")
        st.write(f"Confidence: {layer2_out['confidence']:.3f}")
        st.dataframe(
            pd.DataFrame({
                "Phase": list(layer2_out["phase_probabilities"].keys()),
                "Probability": list(layer2_out["phase_probabilities"].values()),
            }).sort_values("Probability", ascending=False),
            use_container_width=True,
        )

    with c:
        st.markdown("**Fusion**")
        st.write(f"Confidence: {fusion_out['confidence']:.3f}")
        st.dataframe(
            pd.DataFrame({
                "Phase": list(fusion_out["phase_probabilities"].keys()),
                "Probability": list(fusion_out["phase_probabilities"].values()),
            }).sort_values("Probability", ascending=False),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Input summary")
    st.dataframe(
        pd.DataFrame({
            "Field": ["Log date", "Last period start", "Cervical mucus", "Symptoms"],
            "Value": [
                str(log_date),
                str(last_period_start),
                cervical_mucus,
                ", ".join(symptoms_selected) if symptoms_selected else "None",
            ],
        }),
        use_container_width=True,
    )
