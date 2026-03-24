import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta
from pathlib import Path

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
def load_layer2_model():
    model_paths = [
        Path("layer2a_clf_mucus.joblib"),
        Path("models/layer2a_clf_mucus.joblib"),
        Path("artifacts/layer2a_clf_mucus.joblib"),
    ]
    feature_paths = [
        Path("layer2a_feature_cols.joblib"),
        Path("models/layer2a_feature_cols.joblib"),
        Path("artifacts/layer2a_feature_cols.joblib"),
    ]

    model_path = next((p for p in model_paths if p.exists()), None)
    feature_path = next((p for p in feature_paths if p.exists()), None)

    if model_path is None:
        raise FileNotFoundError(
            "Could not find layer2a_clf_mucus.joblib. Upload it to the repo root, models/, or artifacts/."
        )
    if feature_path is None:
        raise FileNotFoundError(
            "Could not find layer2a_feature_cols.joblib. Upload it to the repo root, models/, or artifacts/."
        )

    clf_mucus = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)
    return clf_mucus, feature_cols


def build_layer2_input_row(
    log_date: date,
    last_period_start: date,
    cycle_day: int,
    cycle_length: int,
    mucus_type: str,
    symptoms_selected: list[str],
) -> pd.DataFrame:
    symptom_flags = {s: 1 if s in symptoms_selected else 0 for s in SYMPTOMS}

    mucus_score_map = {
        "unknown": 0,
        "dry": 0,
        "sticky": 1,
        "creamy": 2,
        "watery": 3,
        "eggwhite": 4,
    }

    row = {
        "log_date": pd.to_datetime(log_date),
        "last_period_start": pd.to_datetime(last_period_start),
        "cycle_day": cycle_day,
        "cycle_length": cycle_length,
        "cervical_mucus_estimated_type_final": mucus_type,
        "cervical_mucus_fertility_score_final": mucus_score_map[mucus_type],
        **symptom_flags,
    }

    # placeholders
    for pca_col in ["PC1", "PC2", "PC3", "PC4"]:
        row[pca_col] = 0.0

    # optional engineered placeholders if missing
    for extra_col in [
        "symptom_burden_score",
        "pain_score",
        "recovery_score",
        "mood_score",
        "body_score",
        "digestive_score",
    ]:
        row.setdefault(extra_col, 0.0)

    return pd.DataFrame([row])


def prepare_layer2_input_for_model(layer2_input: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = layer2_input.copy()

    # basic composites from raw checkbox inputs
    def mean_of(cols):
        available = [c for c in cols if c in df.columns]
        if not available:
            return 0.0
        return df[available].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.0)

    if "symptom_burden_score" in feature_cols:
        df["symptom_burden_score"] = mean_of(
            ["headaches", "cramps", "sorebreasts", "fatigue", "sleepissue",
             "moodswing", "stress", "foodcravings", "indigestion", "bloating"]
        )
    if "pain_score" in feature_cols:
        df["pain_score"] = mean_of(["cramps", "headaches", "sorebreasts", "bloating"])
    if "recovery_score" in feature_cols:
        df["recovery_score"] = mean_of(["fatigue", "sleepissue", "stress"])
    if "mood_score" in feature_cols:
        df["mood_score"] = mean_of(["moodswing", "stress", "foodcravings"])
    if "body_score" in feature_cols:
        df["body_score"] = mean_of(["cramps", "sorebreasts", "bloating", "indigestion"])
    if "digestive_score" in feature_cols:
        df["digestive_score"] = mean_of(["bloating", "indigestion", "foodcravings"])

    # standardized mucus helpers if the saved model expects them
    mucus_type = str(df.loc[df.index[0], "cervical_mucus_estimated_type_final"]).strip().lower()
    df["cervical_mucus_type_std"] = mucus_type
    df["cervical_mucus_score"] = pd.to_numeric(
        df["cervical_mucus_fertility_score_final"], errors="coerce"
    ).fillna(0.0)
    df["mucus_fertile_flag"] = 1 if mucus_type in ["watery", "eggwhite"] else 0

    # interaction terms if expected
    interaction_pairs = [
        ("cramps", "bloating"),
        ("sorebreasts", "foodcravings"),
        ("fatigue", "sleepissue"),
        ("stress", "moodswing"),
        ("pain_score", "body_score"),
        ("digestive_score", "mood_score"),
    ]
    for a, b in interaction_pairs:
        col = f"{a}__x__{b}"
        if col in feature_cols:
            a_val = pd.to_numeric(df[a], errors="coerce").fillna(0.0) if a in df.columns else 0.0
            b_val = pd.to_numeric(df[b], errors="coerce").fillna(0.0) if b in df.columns else 0.0
            df[col] = a_val * b_val

    # ensure all required columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # final order
    df = df[feature_cols].copy()

    # numeric cleanup
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def extract_layer2_probs_from_model(clf_mucus, layer2_input_model: pd.DataFrame) -> tuple[str, dict]:
    proba = clf_mucus.predict_proba(layer2_input_model)[0]
    classes = clf_mucus.named_steps["model"].classes_
    layer2_probs = {cls: float(p) for cls, p in zip(classes, proba)}
    layer2_phase = classes[proba.argmax()]
    return layer2_phase, layer2_probs


st.title("InBalance Real Layer 1 + Layer 2 + Fusion")
st.caption("Uses the real saved Layer 2 model files.")

left, right = st.columns([1.15, 1])

with left:
    st.subheader("1) Period history")
    n_periods = st.number_input(
        "How many period starts do you want to enter?",
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
    symptoms_selected = st.multiselect("Choose symptoms for that day", SYMPTOMS, default=[])

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

    period_starts_sorted = sorted(period_starts)
    last_period_start = max(period_starts_sorted)

    # Layer 1
    layer1_out = layer1_period_predictor.get_layer1_output(
        period_starts=period_starts_sorted,
        current_date=log_date,
    )

    prediction = layer1_out.get("prediction")
    phase_prior = layer1_out.get("phase_prior")

    if prediction is None or phase_prior is None:
        st.error("Layer 1 could not generate output. Please enter at least 2 valid period starts.")
        st.stop()

    baseline_next_period = pd.Timestamp(prediction["predicted_start"]).date()
    cycle_length_estimate = int(prediction["estimated_cycle_length"])
    cycle_lengths = layer1_out.get("cycle_lengths", [])
    cycle_day = int(phase_prior["cycle_day"])

    # Layer 2
    clf_mucus, feature_cols = load_layer2_model()

    raw_layer2_input = build_layer2_input_row(
        log_date=log_date,
        last_period_start=last_period_start,
        cycle_day=cycle_day,
        cycle_length=cycle_length_estimate,
        mucus_type=mucus_type,
        symptoms_selected=symptoms_selected,
    )

    model_layer2_input = prepare_layer2_input_for_model(raw_layer2_input, feature_cols)
    layer2_phase, layer2_probs = extract_layer2_probs_from_model(clf_mucus, model_layer2_input)

    # Fusion
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
        st.markdown("**Layer 1 summary**")
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
        st.markdown("**Layer 2 probabilities**")
        st.dataframe(
            pd.DataFrame({
                "Phase": list(layer2_probs.keys()),
                "Probability": list(layer2_probs.values()),
            }).sort_values("Probability", ascending=False),
            use_container_width=True,
        )

    with c:
        st.markdown("**Fusion probabilities**")
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
            "Field": ["Log date", "Cervical mucus", "Symptoms"],
            "Value": [
                str(log_date),
                mucus_type,
                ", ".join(symptoms_selected) if symptoms_selected else "None",
            ],
        }),
        use_container_width=True,
    )
