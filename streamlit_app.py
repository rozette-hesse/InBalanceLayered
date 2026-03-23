import streamlit as st
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List

# --------------------------------------------------
# IMPORT YOUR REAL MODULES
# --------------------------------------------------
from layer1_period_predictor import predict_next_period_layer1
from layer2a_phase_predictor import Layer2APredictor, Layer2AConfig, Layer2AFeatureBuilder
from layer_fusion import (
    LayerFusionConfig,
    predict_with_fusion,
)

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
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

# --------------------------------------------------
# CACHED MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_layer2_model():
    """
    Option A:
    If you saved a trained Layer2APredictor object with pickle/joblib, load it here.

    Option B:
    If you only have code and training data, you can fit it here once and cache it.
    """
    # ---------- OPTION A: load pretrained model ----------
    # import joblib
    # predictor = joblib.load("layer2a_predictor.joblib")
    # return predictor

    # ---------- OPTION B: fit from training file ----------
    training_path = "mcphases_with_estimated_cervical_mucus_v3.csv"
    df_train = pd.read_csv(training_path)

    predictor = Layer2APredictor(
        config=Layer2AConfig(
            target_col="phase",
            group_col="id",
            random_state=42,
            test_size=0.2,
            max_iter=3000,
        ),
        feature_builder=Layer2AFeatureBuilder(
            mucus_type_col="cervical_mucus_estimated_type_final",
            mucus_score_col="cervical_mucus_fertility_score_final",
        ),
    )
    predictor.fit(df_train)
    return predictor


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def build_layer2_input_row(
    log_date: date,
    last_period_start: date,
    cycle_day: int,
    cycle_length: int,
    mucus_type: str,
    symptoms_selected: List[str],
) -> pd.DataFrame:
    """
    Build one-row dataframe using the real feature names expected by Layer 2A.
    """
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

    # If your Layer2A code expects PCA columns but computes missing columns safely, leave them out.
    # If needed, add placeholders:
    for pca_col in ["PC1", "PC2", "PC3", "PC4"]:
        row[pca_col] = 0.0

    return pd.DataFrame([row])


def extract_layer2_probs(pred_df: pd.DataFrame) -> Dict[str, float]:
    probs = {}
    for phase in ["Menstrual", "Follicular", "Fertility", "Luteal"]:
        col = f"prob_{phase}"
        probs[phase] = float(pred_df.iloc[0].get(col, 0.0))
    return probs


def safe_cycle_day(last_period_start: date, current_date: date) -> int:
    return max(1, (current_date - last_period_start).days + 1)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("InBalance Real Layer 1 + Layer 2 + Fusion")
st.caption("Uses your real Python modules, not the demo approximation.")

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

    default_today = date.today()
    period_starts = []
    period_ends = []

    for i in range(int(n_periods)):
        st.markdown(f"**Period {i+1}**")
        c1, c2 = st.columns(2)
        with c1:
            p_start = st.date_input(
                f"Start date #{i+1}",
                value=default_today - timedelta(days=(int(n_periods) - i) * 30),
                key=f"start_{i}",
            )
        with c2:
            p_end = st.date_input(
                f"End date #{i+1}",
                value=default_today - timedelta(days=(int(n_periods) - i) * 30 - 4),
                key=f"end_{i}",
            )
        period_starts.append(p_start)
        period_ends.append(p_end)

    st.subheader("2) Symptom log")
    log_date = st.date_input("Symptom log date", value=default_today)
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

    st.markdown("---")
    run_btn = st.button("Run real prediction", type="primary", use_container_width=True)

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
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

    # ---------- Layer 1 ----------
    # Expected signature:
    # predict_next_period_layer1(period_starts: List[date]) -> dict
    # Example returned keys:
    # {
    #   "predicted_next_period": date,
    #   "cycle_length_estimate": int,
    #   "cycle_lengths": [...]
    # }
    layer1_result = predict_next_period_layer1(period_starts_sorted)

    baseline_next_period = layer1_result["predicted_next_period"]
    cycle_length_estimate = int(layer1_result["cycle_length_estimate"])
    cycle_lengths = layer1_result.get("cycle_lengths", [])

    last_period_start = max(period_starts_sorted)
    cycle_day = safe_cycle_day(last_period_start, log_date)

    # ---------- Layer 2 ----------
    predictor = load_layer2_model()

    layer2_input = build_layer2_input_row(
        log_date=log_date,
        last_period_start=last_period_start,
        cycle_day=cycle_day,
        cycle_length=cycle_length_estimate,
        mucus_type=mucus_type,
        symptoms_selected=symptoms_selected,
    )

    layer2_pred_df = predictor.predict(layer2_input)
    layer2_probs = extract_layer2_probs(layer2_pred_df)

    # ---------- Fusion ----------
    fusion_config = LayerFusionConfig(
        layer1_weight=layer1_weight,
        layer2_weight=layer2_weight,
        menstrual_boost_from_layer2=1.10,
        fertility_boost_from_layer2=1.15,
        luteal_boost_from_layer2=1.00,
        follicular_boost_from_layer2=0.95,
        low_confidence_threshold=0.45,
    )

    fusion_result = predict_with_fusion(
        cycle_day=cycle_day,
        cycle_length=cycle_length_estimate,
        layer2_probs=layer2_probs,
        config=fusion_config,
    )

    shift_days = int(fusion_result["period_shift_days"])
    adjusted_next_period = baseline_next_period + timedelta(days=shift_days)

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Layer 1 baseline next period", str(baseline_next_period))
    c2.metric("Layer 2 predicted phase", str(layer2_pred_df.iloc[0]["predicted_phase"]))
    c3.metric("Fusion predicted phase", str(fusion_result["predicted_phase"]))

    c4, c5, c6 = st.columns(3)
    c4.metric("Estimated cycle length", f"{cycle_length_estimate} days")
    c5.metric("Cycle day on log date", cycle_day)
    c6.metric("Fusion shift", f"{shift_days:+d} day(s)")

    st.metric("Final adjusted next period", str(adjusted_next_period))

    st.markdown("---")
    st.subheader("Layer comparison")

    out1, out2, out3 = st.columns(3)

    with out1:
        st.markdown("**Layer 1 summary**")
        st.write(f"Last period start: {last_period_start}")
        st.write(f"Cycle lengths used: {cycle_lengths}")

    with out2:
        st.markdown("**Layer 2 probabilities**")
        layer2_probs_df = pd.DataFrame(
            {"Phase": list(layer2_probs.keys()), "Probability": list(layer2_probs.values())}
        ).sort_values("Probability", ascending=False)
        st.dataframe(layer2_probs_df, use_container_width=True)

    with out3:
        st.markdown("**Fusion probabilities**")
        fusion_probs_df = pd.DataFrame(
            {
                "Phase": list(fusion_result["phase_probabilities"].keys()),
                "Probability": list(fusion_result["phase_probabilities"].values()),
            }
        ).sort_values("Probability", ascending=False)
        st.dataframe(fusion_probs_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Inputs used")

    input_summary = pd.DataFrame(
        {
            "Field": [
                "Log date",
                "Cervical mucus",
                "Selected symptoms",
            ],
            "Value": [
                str(log_date),
                mucus_type,
                ", ".join(symptoms_selected) if symptoms_selected else "None",
            ],
        }
    )
    st.dataframe(input_summary, use_container_width=True)
