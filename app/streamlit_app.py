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


st.set_page_config(page_title="InBalance Cycle Engine", layout="wide")

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
    return [key for key, severity in symptom_state.items() if severity_to_binary(severity) == 1]


def compute_bleed_lengths(period_rows: list[dict]) -> list[int]:
    lengths = []
    for row in period_rows:
        if row["start"] and row["end"] and row["end"] >= row["start"]:
            lengths.append((row["end"] - row["start"]).days + 1)
    return lengths


def safe_avg(values: list[int]):
    return round(sum(values) / len(values), 2) if values else None


def get_status_color(status: str) -> str:
    mapping = {
        "Red Day": "#ef4444",
        "Light Red Day": "#f97316",
        "Green Day": "#22c55e",
        "Need More Data": "#a78bfa",
    }
    return mapping.get(status, "#64748b")


def get_phase_color(phase: str) -> str:
    mapping = {
        "Menstrual": "#ef4444",
        "Follicular": "#f59e0b",
        "Fertility": "#22c55e",
        "Luteal": "#8b5cf6",
    }
    return mapping.get(phase, "#64748b")


def render_badge(text: str, color: str) -> str:
    return f"""
    <span style="
        display:inline-block;
        padding:0.35rem 0.75rem;
        border-radius:999px;
        background:{color}20;
        color:{color};
        font-weight:600;
        font-size:0.9rem;
        border:1px solid {color}55;
    ">
        {text}
    </span>
    """


def render_card(title: str, value: str, subtitle: str = "") -> str:
    return f"""
    <div style="
        background:white;
        border:1px solid #e5e7eb;
        border-radius:20px;
        padding:18px 20px;
        box-shadow:0 4px 18px rgba(15,23,42,0.05);
        min-height:120px;
    ">
        <div style="font-size:0.9rem;color:#6b7280;margin-bottom:8px;">{title}</div>
        <div style="font-size:1.8rem;font-weight:700;color:#111827;line-height:1.2;">{value}</div>
        <div style="font-size:0.9rem;color:#6b7280;margin-top:8px;">{subtitle}</div>
    </div>
    """


def render_circle(phase: str, cycle_day, fertility_status: str) -> str:
    phase_color = get_phase_color(phase)
    status_color = get_status_color(fertility_status)
    day_text = cycle_day if cycle_day is not None else "?"
    return f"""
    <div style="display:flex;justify-content:center;align-items:center;padding:12px 0 4px 0;">
        <div style="
            width:260px;
            height:260px;
            border-radius:50%;
            background:
                radial-gradient(circle at center, white 52%, transparent 53%),
                conic-gradient(
                    #ef4444 0deg 60deg,
                    #f59e0b 60deg 170deg,
                    #22c55e 170deg 230deg,
                    #8b5cf6 230deg 360deg
                );
            display:flex;
            align-items:center;
            justify-content:center;
            box-shadow:0 10px 30px rgba(0,0,0,0.08);
            border:10px solid #fff;
        ">
            <div style="
                width:180px;
                height:180px;
                border-radius:50%;
                background:#fff;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                text-align:center;
                box-shadow: inset 0 0 0 1px #f1f5f9;
                padding:12px;
            ">
                <div style="font-size:0.85rem;color:#6b7280;">Cycle day</div>
                <div style="font-size:2.4rem;font-weight:800;color:#111827;line-height:1;">{day_text}</div>
                <div style="margin-top:10px;font-size:1.05rem;font-weight:700;color:{phase_color};">{phase}</div>
                <div style="margin-top:10px;">
                    {render_badge(fertility_status, status_color)}
                </div>
            </div>
        </div>
    </div>
    """


def render_section_title(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="margin: 8px 0 14px 0;">
            <div style="font-size:1.2rem;font-weight:800;color:#111827;">{title}</div>
            <div style="font-size:0.95rem;color:#6b7280;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "period_count" not in st.session_state:
    st.session_state.period_count = 3

if "period_defaults" not in st.session_state:
    st.session_state.period_defaults = [
        {"start": date(2026, 1, 1), "end": date(2026, 1, 5)},
        {"start": date(2026, 1, 29), "end": date(2026, 2, 2)},
        {"start": date(2026, 2, 26), "end": date(2026, 3, 1)},
    ]

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fff7fb 0%, #f8fafc 100%);
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(15,23,42,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
        color: white;
        padding: 28px 30px;
        border-radius: 28px;
        box-shadow: 0 14px 40px rgba(139,92,246,0.25);
        margin-bottom: 24px;
    ">
        <div style="font-size:0.95rem;opacity:0.9;margin-bottom:6px;">InBalance</div>
        <div style="font-size:2rem;font-weight:800;line-height:1.1;">Cycle Forecast & Daily Status</div>
        <div style="margin-top:10px;font-size:1rem;opacity:0.95;max-width:720px;">
            Layer 1 gives your cycle forecast, Layer 2 reads today’s body signals, and Layer 3 tells you whether your cycle looks on track or slightly shifted.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_section_title("Period history", "Add past periods so the forecast engine can estimate timing.")

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

render_section_title("Symptoms today", "Choose how strong each symptom feels today.")
symptom_state = {}
s1, s2 = st.columns(2)
symptom_items = list(SYMPTOM_LABELS.items())

for idx, (key, label) in enumerate(symptom_items):
    target_col = s1 if idx % 2 == 0 else s2
    with target_col:
        symptom_state[key] = st.selectbox(
            label,
            SEVERITY_OPTIONS,
            index=0,
            key=f"sym_{key}",
        )

render_section_title("Cervical mucus", "This is one of the strongest clues for fertile timing.")
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

    st.divider()

    left, right = st.columns([1.1, 1.3])

    with left:
        st.markdown(
            render_circle(
                phase=result["final_phase"],
                cycle_day=result["layer1"].get("cycle_day"),
                fertility_status=result["layer2"]["fertility_status"] if result["layer2"] else "Need More Data",
            ),
            unsafe_allow_html=True,
        )

    with right:
        render_section_title("Today at a glance", "Your forecast, current phase, and daily fertility-style status.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                render_card(
                    "Expected next period",
                    result["layer1"].get("predicted_next_period") or "N/A",
                    "Based on cycle history",
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_card(
                    "Forecast confidence",
                    result["layer1"].get("forecast_confidence", "N/A").title(),
                    f"Regularity: {result['layer1'].get('regularity_status', 'N/A').replace('_', ' ')}",
                ),
                unsafe_allow_html=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(
                render_card(
                    "Possible ovulation date",
                    result["layer1"].get("possible_ovulation_date") or "N/A",
                    "Forecast estimate",
                ),
                unsafe_allow_html=True,
            )
        with c4:
            fertility_status = result["layer2"]["fertility_status"] if result["layer2"] else "Need More Data"
            st.markdown(
                render_card(
                    "Daily status",
                    fertility_status,
                    "Symptom-informed status",
                ),
                unsafe_allow_html=True,
            )

    render_section_title("Forecast & daily interpretation")

    a, b, c = st.columns(3)
    with a:
        st.metric("Cycle day", result["layer1"].get("cycle_day"))
        st.metric("Estimated cycle length", result["layer1"].get("estimated_cycle_length"))
        st.metric("Average bleed length", avg_bleed_length if avg_bleed_length is not None else "N/A")

    with b:
        st.markdown("**Fertile window**")
        fw = result["layer1"].get("fertile_window")
        if fw:
            st.success(f"{fw['start']} → {fw['end']}")
        else:
            st.info("Not enough history yet")

        st.markdown("**Next period window**")
        npw = result["layer1"].get("next_period_window")
        if npw:
            st.warning(f"{npw['start']} → {npw['end']}")
        else:
            st.info("Not enough history yet")

    with c:
        if result["layer2"] is not None:
            st.markdown("**Layer 2 status**")
            status_color = get_status_color(result["layer2"]["fertility_status"])
            st.markdown(
                render_badge(result["layer2"]["fertility_status"], status_color),
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(f"**Symptom phase:** {result['layer2']['top_phase']}")
            st.write(f"**Signal confidence:** {result['layer2']['signal_confidence'].title()}")

        if result["layer3"] is not None:
            st.write("")
            st.markdown("**Timing interpretation**")
            st.info(result["layer3"]["timing_status"])
            st.caption(result["layer3"]["timing_note"])

    if result["layer2"] is not None:
        render_section_title("Why the app said this")
        for line in result["layer2"]["explanations"]:
            st.write(f"- {line}")

    render_section_title("Phase probabilities")
    probs = result["final_phase_probs"]
    for phase, value in probs.items():
        st.write(f"**{phase}**")
        st.progress(float(value))
        st.caption(f"{round(value * 100, 1)}%")

    render_section_title("Recommendations")
    recs = get_recommendations(result["final_phase"])
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            render_card("Workout", recs.get("workout", ""), "Based on your current likely phase"),
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            render_card("Nutrition", recs.get("nutrition", ""), "Based on your current likely phase"),
            unsafe_allow_html=True,
        )

    with st.expander("Raw outputs"):
        st.json(result)
else:
    st.info("Fill in your period history and symptoms above, then click Run prediction.")
