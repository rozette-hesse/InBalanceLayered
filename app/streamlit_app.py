import sys
from pathlib import Path
from datetime import date

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from engine.config import MUCUS_OPTIONS
from engine.layer_fusion import get_fused_output
from engine.recommender import get_recommendations


st.set_page_config(page_title="InBalance Cycle Engine", layout="centered")

SYMPTOM_LABELS = {
    "headaches": "Headaches",
    "cramps": "Cramps",
    "sorebreasts": "Breast tenderness",
    "fatigue": "Fatigue",
    "sleepissue": "Sleep issues",
    "moodswing": "Mood swings",
    "stress": "Stress",
    "foodcravings": "Food cravings",
    "indigestion": "Indigestion",
    "bloating": "Bloating",
}

SEVERITY_OPTIONS = ["None", "Light", "Moderate", "Heavy"]


def severity_to_binary(value: str) -> int:
    return 0 if value == "None" else 1


def parse_selected_symptoms(symptom_state: dict) -> list[str]:
    selected = []
    for key, severity in symptom_state.items():
        if severity_to_binary(severity) == 1:
            selected.append(key)
    return selected


def compute_bleed_lengths(period_rows: list[dict]) -> list[int]:
    lengths = []
    for row in period_rows:
        if row["start"] and row["end"] and row["end"] >= row["start"]:
            lengths.append((row["end"] - row["start"]).days + 1)
    return lengths


def safe_avg(values: list[int]):
    return round(sum(values) / len(values), 2) if values else None


if "period_count" not in st.session_state:
    st.session_state.period_count = 3

if "period_defaults" not in st.session_state:
    st.session_state.period_defaults = [
        {"start": date(2026, 1, 1), "end": date(2026, 1, 5)},
        {"start": date(2026, 1, 29), "end": date(2026, 2, 2)},
        {"start": date(2026, 2, 26), "end": date(2026, 3, 1)},
    ]

st.title("InBalance Cycle Engine")
st.write("Layer 1 = period history, Layer 2 = trained symptom model, Fusion = final output")

st.subheader("1) Period history")

col_add, col_remove = st.columns(2)
with col_add:
    if st.button("Add another period"):
        st.session_state.period_count += 1
with col_remove:
    if st.button("Remove last period") and st.session_state.period_count > 1:
        st.session_state.period_count -= 1

period_rows = []

for i in range(st.session_state.period_count):
    st.markdown(f"**Period {i + 1}**")
    c1, c2 = st.columns(2)

    default_start = None
    default_end = None
    if i < len(st.session_state.period_defaults):
        default_start = st.session_state.period_defaults[i]["start"]
        default_end = st.session_state.period_defaults[i]["end"]

    with c1:
        start_date = st.date_input(
            f"Start date #{i + 1}",
            value=default_start if default_start else date.today(),
            key=f"start_{i}",
        )
    with c2:
        end_date = st.date_input(
            f"End date #{i + 1}",
            value=default_end if default_end else start_date,
            key=f"end_{i}",
        )

    period_rows.append({"start": start_date, "end": end_date})

st.subheader("2) Symptoms today")
st.caption("Choose how strong each symptom feels today.")

symptom_state = {}
for key, label in SYMPTOM_LABELS.items():
    symptom_state[key] = st.selectbox(
        label,
        SEVERITY_OPTIONS,
        index=0,
        key=f"sym_{key}",
    )

st.subheader("3) Cervical mucus")
cervical_mucus = st.selectbox(
    "Select cervical mucus type",
    MUCUS_OPTIONS,
    index=MUCUS_OPTIONS.index("unknown") if "unknown" in MUCUS_OPTIONS else 0,
)

st.subheader("4) Run prediction")
run = st.button("Run prediction", type="primary")

if run:
    invalid_rows = [i + 1 for i, row in enumerate(period_rows) if row["end"] < row["start"]]
    if invalid_rows:
        st.error(f"End date cannot be before start date for period(s): {', '.join(map(str, invalid_rows))}")
        st.stop()

    period_starts = [row["start"].strftime("%Y-%m-%d") for row in period_rows]
    selected_symptoms = parse_selected_symptoms(symptom_state)

    result = get_fused_output(
        period_starts=period_starts,
        symptoms=selected_symptoms,
        cervical_mucus=cervical_mucus,
    )

    bleed_lengths = compute_bleed_lengths(period_rows)
    avg_bleed_length = safe_avg(bleed_lengths)

    layer1_top = max(result["layer1"]["phase_probs"], key=result["layer1"]["phase_probs"].get)
    layer2_top = result["layer2"]["top_phase"] if result["layer2"] else None
    final_top = result["final_phase"]

    st.subheader("Prediction summary")
    a, b, c = st.columns(3)
    with a:
        st.metric("Cycle day today", result["layer1"].get("cycle_day"))
    with b:
        st.metric("Predicted next period", result["layer1"].get("predicted_next_period"))
    with c:
        st.metric("Final predicted phase", final_top)

    d, e = st.columns(2)
    with d:
        st.metric("Estimated cycle length", result["layer1"].get("estimated_cycle_length"))
    with e:
        st.metric("Average bleed length", avg_bleed_length if avg_bleed_length is not None else "N/A")

    st.subheader("How the prediction changed")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Layer 1**\n\n{layer1_top}")
    with col2:
        if layer2_top:
            st.info(f"**Layer 2**\n\n{layer2_top}")
        else:
            st.info("**Layer 2**\n\nNo symptom input")
    with col3:
        st.success(f"**Final**\n\n{final_top}")

    if result["layer2"] is not None:
        st.write(
            f"Based on period history alone, the model leans toward **{layer1_top}**. "
            f"After adding symptoms and cervical mucus, Layer 2 leans toward **{layer2_top}**. "
            f"The final fused prediction is **{final_top}**."
        )
    else:
        st.write(
            f"No symptoms or mucus were added, so the final prediction stays based on period history: **{final_top}**."
        )

    st.subheader("Final phase probabilities")
    final_probs = result["final_phase_probs"]
    prob_df = {
        "Phase": list(final_probs.keys()),
        "Probability": [round(v, 4) for v in final_probs.values()],
    }
    st.dataframe(prob_df, use_container_width=True)

    st.subheader("Layer details")
    with st.expander("Layer 1 output"):
        st.json(result["layer1"])

    if result["layer2"] is not None:
        with st.expander("Layer 2 output"):
            st.json(result["layer2"])

    st.subheader("Recommendations")
    st.json(get_recommendations(final_top))
