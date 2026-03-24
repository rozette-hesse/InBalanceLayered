import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date, timedelta

from engine import layer1_period_predictor
from engine import layer_fusion


st.set_page_config(page_title="InBalance Real Cycle Engine", layout="wide")

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


@st.cache_resource
def load_layer2_assets():
    model_candidates = [
        Path("layer2a_clf_mucus.joblib"),
        Path("models/layer2a_clf_mucus.joblib"),
        Path("artifacts/layer2a_clf_mucus.joblib"),
    ]
    feature_candidates = [
        Path("layer2a_feature_cols.joblib"),
        Path("models/layer2a_feature_cols.joblib"),
        Path("artifacts/layer2a_feature_cols.joblib"),
    ]

    model_path = next((p for p in model_candidates if p.exists()), None)
    feature_path = next((p for p in feature_candidates if p.exists()), None)

    if model_path is None:
        raise FileNotFoundError("Could not find layer2a_clf_mucus.joblib")
    if feature_path is None:
        raise FileNotFoundError("Could not find layer2a_feature_cols.joblib")

    clf_mucus = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    feature_cols = [str(c) for c in feature_cols]
    return clf_mucus, feature_cols


def build_raw_layer2_row(
    log_date: date,
    last_period_start: date,
    cycle_day: int,
    cycle_length: int,
    mucus_type: str,
    symptoms_selected: list[str],
) -> dict:
    mucus_score_map = {
        "unknown": 0.0,
        "dry": 0.0,
        "sticky": 1.0,
        "creamy": 2.0,
        "watery": 3.0,
        "eggwhite": 4.0,
    }

    row = {
        "log_date": pd.to_datetime(log_date),
        "last_period_start": pd.to_datetime(last_period_start),
        "cycle_day": float(cycle_day),
        "cycle_length": float(cycle_length),
        "cervical_mucus_estimated_type_final": mucus_type,
        "cervical_mucus_fertility_score_final": mucus_score_map[mucus_type],
    }

    for s in SYMPTOMS:
        row[s] = 1.0 if s in symptoms_selected else 0.0

    return row


def mean_safe(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    return float(arr.mean())


def prepare_layer2_input_for_model(raw_row: dict, feature_cols: list[str]) -> pd.DataFrame:
    x = {col: 0.0 for col in feature_cols}

    # raw symptoms
    for s in SYMPTOMS:
        if s in x:
            x[s] = float(raw_row.get(s, 0.0))

    # direct cycle timing if expected
    for col in ["cycle_day", "cycle_length"]:
        if col in x:
            x[col] = float(raw_row.get(col, 0.0))

    # PCA placeholders
    for col in ["PC1", "PC2", "PC3", "PC4"]:
        if col in x:
            x[col] = 0.0

    # mucus helpers
    mucus_type = str(raw_row.get("cervical_mucus_estimated_type_final", "unknown")).strip().lower()
    mucus_score = float(raw_row.get("cervical_mucus_fertility_score_final", 0.0))
    mucus_fertile_flag = 1.0 if mucus_type in ["watery", "eggwhite"] else 0.0

    if "cervical_mucus_score" in x:
        x["cervical_mucus_score"] = mucus_score
    if "mucus_fertile_flag" in x:
        x["mucus_fertile_flag"] = mucus_fertile_flag

    # composite scores
    symptom_burden_score = mean_safe([raw_row[s] for s in [
        "headaches", "cramps", "sorebreasts", "fatigue", "sleepissue",
        "moodswing", "stress", "foodcravings", "indigestion", "bloating"
    ]])

    pain_score = mean_safe([raw_row[s] for s in [
        "cramps", "headaches", "sorebreasts", "bloating"
    ]])

    recovery_score = mean_safe([raw_row[s] for s in [
        "fatigue", "sleepissue", "stress"
    ]])

    mood_score = mean_safe([raw_row[s] for s in [
        "moodswing", "stress", "foodcravings"
    ]])

    body_score = mean_safe([raw_row[s] for s in [
        "cramps", "sorebreasts", "bloating", "indigestion"
    ]])

    digestive_score = mean_safe([raw_row[s] for s in [
        "bloating", "indigestion", "foodcravings"
    ]])

    composites = {
        "symptom_burden_score": symptom_burden_score,
        "pain_score": pain_score,
        "recovery_score": recovery_score,
        "mood_score": mood_score,
        "body_score": body_score,
        "digestive_score": digestive_score,
    }

    for k, v in composites.items():
        if k in x:
            x[k] = float(v)

    # interaction terms
    interactions = {
        "cramps__x__bloating": raw_row["cramps"] * raw_row["bloating"],
        "sorebreasts__x__foodcravings": raw_row["sorebreasts"] * raw_row["foodcravings"],
        "fatigue__x__sleepissue": raw_row["fatigue"] * raw_row["sleepissue"],
        "stress__x__moodswing": raw_row["stress"] * raw_row["moodswing"],
        "pain_score__x__body_score": pain_score * body_score,
        "digestive_score__x__mood_score": digestive_score * mood_score,
    }

    for k, v in interactions.items():
        if k in x:
            x[k] = float(v)

    df = pd.DataFrame([x], columns=feature_cols)

    # final numeric coercion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def extract_layer2_prediction(clf_mucus, layer2_input_df: pd.DataFrame):
    proba = clf_mucus.predict_proba(layer2_input_df)[0]
    classes = list(clf_mucus.named_steps["model"].classes_)
    probs = {cls: float(p) for cls, p in zip(classes, proba)}
    pred_phase = classes[int(np.argmax(proba))]
    confidence = float(np.max(proba))
    return pred_phase, confidence, probs


st.title("InBalance Real Layer 1 + Layer 2 + Fusion")
st.caption("Baseline next-period prediction from Layer 1, adjusted using real Layer 2 symptoms + mucus and the fusion layer.")

left, right = st.columns([1.15, 1])

with left:
    st.subheader("1) Period history")
    n_periods = st.number_input(
        "How many periods do you want to enter?",
        min_value=2,
        max_value=12,
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
    mucus_type = st.selectbox("Cervical mucus", MUCUS_OPTIONS, index=0)
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
    run_btn = st.button("Run real prediction", type="primary", use_container_width=True)

if run_btn:
    errors = []
    for i, (s, e) in enumerate(zip(period_starts, period_ends), start=1):
        if e < s:
            errors.append(f"Period {i}: end date is before start date.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    try:
        clf_mucus, feature_cols = load_layer2_assets()
    except Exception as e:
        st.error(f"Failed to load Layer 2 assets: {e}")
        st.stop()

    period_starts_sorted = sorted(period_starts)
    last_period_start = max(period_starts_sorted)

    try:
        layer1_out = layer1_period_predictor.get_layer1_output(
            period_starts=period_starts_sorted,
            current_date=log_date,
        )
    except Exception as e:
        st.error(f"Layer 1 failed: {e}")
        st.stop()

    prediction = layer1_out.get("prediction")
    phase_prior = layer1_out.get("phase_prior")

    if prediction is None or phase_prior is None:
        st.error("Layer 1 could not generate output. Enter at least 2 valid period starts.")
        st.stop()

    baseline_next_period = pd.Timestamp(prediction["predicted_start"]).date()
    cycle_length_estimate = int(prediction["estimated_cycle_length"])
    cycle_lengths = layer1_out.get("cycle_lengths", [])
    cycle_day = int(phase_prior["cycle_day"])

    raw_row = build_raw_layer2_row(
        log_date=log_date,
        last_period_start=last_period_start,
        cycle_day=cycle_day,
        cycle_length=cycle_length_estimate,
        mucus_type=mucus_type,
        symptoms_selected=symptoms_selected,
    )

    layer2_input_df = prepare_layer2_input_for_model(raw_row, feature_cols)

    try:
        layer2_phase, layer2_confidence, layer2_probs = extract_layer2_prediction(
            clf_mucus,
            layer2_input_df,
        )
    except Exception as e:
        st.error(f"Layer 2 prediction failed: {e}")
        st.write("Feature columns expected:", feature_cols)
        st.write("Prepared one-row input:")
        st.dataframe(layer2_input_df, use_container_width=True)
        st.stop()

    try:
        fusion_config = layer_fusion.LayerFusionConfig(
            layer1_weight=layer1_weight,
            layer2_weight=layer2_weight,
            menstrual_boost_from_layer2=1.10,
            fertility_boost_from_layer2=1.15,
            luteal_boost_from_layer2=1.00,
            follicular_boost_from_layer2=0.95,
            low_confidence_threshold=0.45,
        )

        fusion_result = layer_fusion.predict_with_fusion(
            cycle_day=cycle_day,
            cycle_length=cycle_length_estimate,
            layer2_probs=layer2_probs,
            config=fusion_config,
        )
    except Exception as e:
        st.error(f"Fusion failed: {e}")
        st.stop()

    shift_days = int(fusion_result["period_shift_days"])
    adjusted_next_period = baseline_next_period + timedelta(days=shift_days)

    st.markdown("---")
    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Layer 1 baseline next period", str(baseline_next_period))
    c2.metric("Layer 2 predicted phase", str(layer2_phase))
    c3.metric("Fusion predicted phase", str(fusion_result["predicted_phase"]))

    c4, c5, c6 = st.columns(3)
    c4.metric("Estimated cycle length", f"{cycle_length_estimate} days")
    c5.metric("Cycle day on log date", cycle_day)
    c6.metric("Fusion shift", f"{shift_days:+d} day(s)")

    st.metric("Final adjusted next period", str(adjusted_next_period))

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
            }),
            use_container_width=True,
        )

    with b:
        st.markdown("**Layer 2**")
        st.write(f"Confidence: {layer2_confidence:.3f}")
        st.dataframe(
            pd.DataFrame({
                "Phase": list(layer2_probs.keys()),
                "Probability": list(layer2_probs.values()),
            }).sort_values("Probability", ascending=False),
            use_container_width=True,
        )

    with c:
        st.markdown("**Fusion**")
        st.write(f"Confidence: {fusion_result['confidence']:.3f}")
        st.dataframe(
            pd.DataFrame({
                "Phase": list(fusion_result["phase_probabilities"].keys()),
                "Probability": list(fusion_result["phase_probabilities"].values()),
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
                mucus_type,
                ", ".join(symptoms_selected) if symptoms_selected else "None",
            ],
        }),
        use_container_width=True,
    )

    with st.expander("Debug: prepared Layer 2 input row"):
        st.dataframe(layer2_input_df, use_container_width=True)
