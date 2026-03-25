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
st.write("Layer 1 = forecast engine, Layer 2 = daily status engine, Layer 3 = timing consistency")

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

    st.subheader("Today at a glance")
    a, b, c, d = st.columns(4)
    with a:
        st.metric("Cycle day", result["layer1"].get("cycle_day"))
    with b:
        st.metric("Current phase", result["final_phase"])
    with c:
        next_period_value = result["layer1"].get("predicted_next_period") or "N/A"
        st.metric("Expected next period", next_period_value)
    with d:
        fert_status = result["layer2"]["fertility_status"] if result["layer2"] else "Need More Data"
        st.metric("Daily status", fert_status)

    st.subheader("Layer 1 — Forecast")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Estimated cycle length", result["layer1"].get("estimated_cycle_length"))
    with c2:
        st.metric("Average bleed length", avg_bleed_length if avg_bleed_length is not None else "N/A")
    with c3:
        st.metric("Forecast confidence", result["layer1"].get("forecast_confidence"))

    st.write("**Possible ovulation date:**", result["layer1"].get("possible_ovulation_date"))
    st.write("**Fertile window:**", result["layer1"].get("fertile_window"))
    st.write("**Next period window:**", result["layer1"].get("next_period_window"))
    st.write("**Regularity:**", result["layer1"].get("regularity_status"))

    if result["layer2"] is not None:
        st.subheader("Layer 2 — Daily status")
        x, y, z = st.columns(3)
        with x:
            st.metric("Fertility status", result["layer2"]["fertility_status"])
        with y:
            st.metric("Symptom phase", result["layer2"]["top_phase"])
        with z:
            st.metric("Signal confidence", result["layer2"]["signal_confidence"])

        st.write("**Why:**")
        for line in result["layer2"]["explanations"]:
            st.write(f"- {line}")

    if result["layer3"] is not None:
        st.subheader("Layer 3 — Timing interpretation")
        st.metric("Timing status", result["layer3"]["timing_status"])
        st.write(result["layer3"]["timing_note"])

    st.subheader("Final phase probabilities")
    final_probs = result["final_phase_probs"]
    prob_df = {
        "Phase": list(final_probs.keys()),
        "Probability": [round(v, 4) for v in final_probs.values()],
    }
    st.dataframe(prob_df, use_container_width=True)

    st.subheader("Recommendations")
    st.json(get_recommendations(result["final_phase"]))

    with st.expander("Raw outputs"):
        st.json(result)
