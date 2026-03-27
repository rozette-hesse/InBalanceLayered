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

st.set_page_config(page_title="InBalance", layout="centered")

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
    return [k for k, v in symptom_state.items() if severity_to_binary(v) == 1]


def compute_bleed_lengths(period_rows: list[dict]) -> list[int]:
    lengths = []
    for row in period_rows:
        if row["start"] and row["end"] and row["end"] >= row["start"]:
            lengths.append((row["end"] - row["start"]).days + 1)
    return lengths


def safe_avg(values: list[int]):
    return round(sum(values) / len(values), 2) if values else None


def status_color(status: str) -> str:
    if status == "Red Day":
        return "#ef4444"
    if status == "Light Red Day":
        return "#f97316"
    if status == "Green Day":
        return "#22c55e"
    return "#64748b"


def phase_color(phase: str) -> str:
    mapping = {
        "Menstrual": "#ef4444",
        "Follicular": "#f59e0b",
        "Fertility": "#22c55e",
        "Luteal": "#8b5cf6",
    }
    return mapping.get(phase, "#64748b")


def badge(text: str, color: str) -> str:
    return f"""
    <div style="
        display:inline-block;
        padding:8px 16px;
        border-radius:999px;
        background:{color}18;
        color:{color};
        font-weight:700;
        border:1px solid {color}55;
        font-size:16px;
    ">{text}</div>
    """


def render_card(title: str, value: str, subtitle: str = "") -> str:
    return f"""
    <div style="
        background:white;
        border:1px solid #e5e7eb;
        border-radius:20px;
        padding:20px 18px;
        box-shadow:0 4px 18px rgba(15,23,42,0.05);
        min-height:145px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        overflow-wrap:anywhere;
        word-break:break-word;
    ">
        <div style="font-size:0.95rem;color:#6b7280;margin-bottom:10px;">{title}</div>
        <div style="font-size:1.25rem;font-weight:700;color:#111827;line-height:1.35;">{value}</div>
        <div style="font-size:0.9rem;color:#6b7280;margin-top:10px;">{subtitle}</div>
    </div>
    """


def hero_card(cycle_day, phase, daily_status, next_period, next_period_window_text, timing_note) -> str:
    p_color = phase_color(phase)
    s_color = status_color(daily_status)
    return f"""
    <div style="
        background:white;
        border:1px solid #e5e7eb;
        border-radius:28px;
        padding:28px;
        box-shadow:0 8px 24px rgba(15,23,42,0.06);
        margin-bottom:18px;
    ">
        <div style="font-size:14px;color:#6b7280;margin-bottom:8px;">Today</div>
        <div style="font-size:42px;font-weight:800;color:#111827;line-height:1;">Cycle day {cycle_day}</div>
        <div style="font-size:26px;font-weight:700;color:{p_color};margin-top:10px;">{phase}</div>
        <div style="margin-top:14px;">{badge(daily_status, s_color)}</div>

        <div style="margin-top:22px;font-size:18px;color:#111827;">
            <strong>Next period:</strong> {next_period if next_period else "N/A"}
        </div>
        <div style="margin-top:6px;font-size:15px;color:#6b7280;">
            <strong>Expected window:</strong> {next_period_window_text if next_period_window_text else "N/A"}
        </div>

        <div style="margin-top:10px;font-size:16px;color:#6b7280;">
            <strong>Timing:</strong> {timing_note}
        </div>
    </div>
    """


if "period_count" not in st.session_state:
    st.session_state.period_count = 3

if "period_defaults" not in st.session_state:
    st.session_state.period_defaults = [
        {"start": date(2025, 11, 1), "end": date(2025, 11, 5)},
        {"start": date(2025, 12, 17), "end": date(2025, 12, 22)},
        {"start": date(2026, 1, 25), "end": date(2026, 1, 30)},
    ]

st.markdown(
    """
    <style>
    .stApp {
        background: #faf7fb;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 12px 14px;
        border-radius: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("InBalance")
st.caption("Simple cycle forecast + daily status")

st.subheader("Period history")

c1, c2 = st.columns(2)
with c1:
    if st.button("Add period"):
        st.session_state.period_count += 1
with c2:
    if st.button("Remove last") and st.session_state.period_count > 1:
        st.session_state.period_count -= 1

period_rows = []
for i in range(st.session_state.period_count):
    a, b = st.columns(2)

    default_start = None
    default_end = None
    if i < len(st.session_state.period_defaults):
        default_start = st.session_state.period_defaults[i]["start"]
        default_end = st.session_state.period_defaults[i]["end"]

    with a:
        start_date = st.date_input(
            f"Start date #{i+1}",
            value=default_start if default_start else date.today(),
            key=f"start_{i}",
        )
    with b:
        end_date = st.date_input(
            f"End date #{i+1}",
            value=default_end if default_end else start_date,
            key=f"end_{i}",
        )

    period_rows.append({"start": start_date, "end": end_date})

st.subheader("Symptoms today")
symptom_state = {}
for key, label in SYMPTOM_LABELS.items():
    symptom_state[key] = st.selectbox(label, SEVERITY_OPTIONS, index=0, key=f"sym_{key}")

st.subheader("Cervical mucus")
cervical_mucus = st.selectbox(
    "Select cervical mucus type",
    MUCUS_OPTIONS,
    index=MUCUS_OPTIONS.index("unknown") if "unknown" in MUCUS_OPTIONS else 0,
)

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

    next_period = result["layer1"].get("predicted_next_period")
    cycle_day = result["layer1"].get("cycle_day")
    phase = result["final_phase"]
    daily_status = result["layer2"]["fertility_status"] if result["layer2"] else "Need More Data"
    timing_note = result["layer3"]["timing_status"] if result["layer3"] else "Based on cycle history only"

    next_period_window = result["layer1"].get("next_period_window")
    next_period_window_text = (
        f"{next_period_window['start']} to {next_period_window['end']}"
        if next_period_window else "N/A"
    )

    st.markdown(
        hero_card(
            cycle_day=cycle_day,
            phase=phase,
            daily_status=daily_status,
            next_period=next_period,
            next_period_window_text=next_period_window_text,
            timing_note=timing_note,
        ),
        unsafe_allow_html=True,
    )

    ovulation_date = result["layer1"].get("possible_ovulation_date") or "N/A"
    fertile_window = result["layer1"].get("fertile_window")
    fertile_window_text = (
        f"{fertile_window['start']} to {fertile_window['end']}"
        if fertile_window else "N/A"
    )
    forecast_confidence = result["layer1"].get("forecast_confidence", "N/A").title()

    x, y, z = st.columns([1.1, 1.4, 1.0])
    with x:
        st.markdown(
            render_card(
                "Estimated ovulation",
                ovulation_date,
                "Forecast estimate",
            ),
            unsafe_allow_html=True,
        )
    with y:
        st.markdown(
            render_card(
                "Fertile window",
                fertile_window_text,
                "Best estimate from cycle history",
            ),
            unsafe_allow_html=True,
        )
    with z:
        st.markdown(
            render_card(
                "Forecast confidence",
                forecast_confidence,
                "How stable your cycle timing looks",
            ),
            unsafe_allow_html=True,
        )

    explain = ""
    if result["layer3"] is not None:
        explain = result["layer3"]["timing_note"]
    elif result["layer2"] is not None:
        explain = f"Today’s symptoms suggest a {result['layer2']['top_phase'].lower()} pattern."
    else:
        explain = "Prediction is based on cycle history only."

    st.write(explain)

    recs = get_recommendations(phase)
    a, b = st.columns(2)
    with a:
        st.markdown("**Workout**")
        st.write(recs.get("workout", ""))
    with b:
        st.markdown("**Nutrition**")
        st.write(recs.get("nutrition", ""))

    with st.expander("See more details"):
        st.write("**Forecast details**")
        st.write("Cycle day:", result["layer1"].get("cycle_day"))
        st.write("Estimated cycle length:", result["layer1"].get("estimated_cycle_length"))
        st.write("Average bleed length:", avg_bleed_length if avg_bleed_length is not None else "N/A")
        st.write("Possible ovulation date:", result["layer1"].get("possible_ovulation_date"))
        st.write("Fertile window:", result["layer1"].get("fertile_window"))
        st.write("Next period window:", result["layer1"].get("next_period_window"))
        st.write("Regularity:", result["layer1"].get("regularity_status"))
        st.write("Forecast confidence:", result["layer1"].get("forecast_confidence"))

        if result["layer2"] is not None:
            st.write("**Daily status details**")
            st.write("Fertility status:", result["layer2"]["fertility_status"])
            st.write("Symptom phase:", result["layer2"]["top_phase"])
            st.write("Signal confidence:", result["layer2"]["signal_confidence"])
            for line in result["layer2"]["explanations"]:
                st.write("-", line)

        st.write("**Phase probabilities**")
        for phase_name, value in result["final_phase_probs"].items():
            st.write(f"{phase_name}: {round(value * 100, 1)}%")

else:
    st.info("Fill in your period history and today’s symptoms, then click Run prediction.")
