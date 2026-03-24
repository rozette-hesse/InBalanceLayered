import streamlit as st
import pandas as pd
from datetime import date, timedelta

st.set_page_config(page_title="InBalance Next Period Predictor", layout="wide")

SYMPTOMS = [
    "cramps",
    "bloating",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "headaches",
    "indigestion",
]

MUCUS_OPTIONS = ["unknown", "dry", "sticky", "creamy", "watery", "eggwhite"]


# =========================================================
# Layer 1
# =========================================================
def compute_cycle_lengths(period_starts):
    period_starts = sorted(period_starts)
    lengths = []
    for i in range(1, len(period_starts)):
        lengths.append((period_starts[i] - period_starts[i - 1]).days)
    return lengths


def weighted_average(values):
    if not values:
        return None
    weights = list(range(1, len(values) + 1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def layer1_predict_next_period(period_starts):
    period_starts = sorted(period_starts)
    if len(period_starts) < 2:
        return None, None, None

    cycle_lengths = compute_cycle_lengths(period_starts)
    avg_cycle = round(weighted_average(cycle_lengths))
    last_start = period_starts[-1]
    baseline_next = last_start + timedelta(days=avg_cycle)

    return baseline_next, avg_cycle, cycle_lengths


# =========================================================
# Layer 2
# =========================================================
def estimate_cycle_day(last_period_start, today):
    return (today - last_period_start).days + 1


def expected_phase_from_cycle_day(cycle_day, cycle_length=28):
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


def symptom_shift_logic(cycle_day, cycle_length, symptoms, mucus):
    expected_phase = expected_phase_from_cycle_day(cycle_day, cycle_length)
    shift_days = 0
    reasons = []

    has = lambda s: s in symptoms

    # strong menstrual-like signals -> period may come sooner
    if has("cramps"):
        shift_days -= 1
        reasons.append("Cramps suggest stronger menstrual-like timing.")

    if has("cramps") and has("bloating"):
        shift_days -= 1
        reasons.append("Cramps + bloating increase early-period likelihood.")

    # luteal / PMS build-up
    luteal_score = 0
    for s in ["sorebreasts", "bloating", "foodcravings", "moodswing", "fatigue", "sleepissue"]:
        if has(s):
            luteal_score += 1

    if luteal_score >= 3:
        reasons.append("Symptoms are consistent with a luteal/PMS pattern.")

    # mucus-driven fertility / delayed ovulation logic
    if mucus in ["watery", "eggwhite"]:
        if cycle_day >= 12:
            shift_days += 2
            reasons.append("Fertile cervical mucus suggests ovulation may still be active/later.")
        else:
            shift_days += 1
            reasons.append("Fertile cervical mucus slightly delays next-period expectation.")

    elif mucus == "creamy":
        shift_days += 1
        reasons.append("Creamy mucus slightly supports approaching fertility rather than late luteal timing.")

    # if clearly luteal by day timing and PMS symptoms present, keep near baseline
    if expected_phase == "Luteal" and luteal_score >= 2:
        reasons.append("Expected phase and symptoms are aligned, so only a small shift is applied.")

    # extra guardrails
    shift_days = max(-3, min(5, shift_days))

    return shift_days, expected_phase, reasons


# =========================================================
# UI
# =========================================================
st.title("InBalance Next Period Shift Demo")
st.caption("Enter period history, log symptoms for a chosen day, and see how Layer 2 shifts the next-period prediction.")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("1) Period history")

    n_periods = st.number_input("How many periods do you want to enter?", min_value=2, max_value=12, value=3, step=1)

    period_rows = []
    today_default = date.today()

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
        period_rows.append({"start": start, "end": end})

    period_df = pd.DataFrame(period_rows)

    st.subheader("2) Symptom log for a chosen day")

    log_date = st.date_input("Symptom log date", value=today_default, key="log_date")
    mucus = st.selectbox("Cervical mucus", MUCUS_OPTIONS, index=0)

    selected_symptoms = st.multiselect(
        "Choose symptoms for that day",
        SYMPTOMS,
        default=[]
    )

with right:
    st.subheader("3) Prediction")

    valid_periods = True
    starts = []
    errors = []

    for idx, row in period_df.iterrows():
        s = row["start"]
        e = row["end"]
        if e < s:
            valid_periods = False
            errors.append(f"Period {idx+1}: end date is before start date.")
        starts.append(s)

    starts = sorted(starts)

    if not valid_periods:
        for err in errors:
            st.error(err)
    else:
        baseline_next, avg_cycle, cycle_lengths = layer1_predict_next_period(starts)

        if baseline_next is None:
            st.warning("Please enter at least 2 period start dates.")
        else:
            last_start = starts[-1]
            cycle_day = estimate_cycle_day(last_start, log_date)
            shift_days, expected_phase, reasons = symptom_shift_logic(
                cycle_day=cycle_day,
                cycle_length=avg_cycle,
                symptoms=selected_symptoms,
                mucus=mucus,
            )
            adjusted_next = baseline_next + timedelta(days=shift_days)

            st.metric("Layer 1 baseline next period", baseline_next.strftime("%Y-%m-%d"))
            st.metric("Average cycle length", f"{avg_cycle} days")
            st.metric("Cycle day on symptom log date", cycle_day)
            st.metric("Expected phase from timing", expected_phase)
            st.metric("Layer 2 shift", f"{shift_days:+d} day(s)")
            st.metric("Final adjusted next period", adjusted_next.strftime("%Y-%m-%d"))

            st.markdown("---")
            st.subheader("Explanation")

            if reasons:
                for r in reasons:
                    st.write(f"- {r}")
            else:
                st.write("- No strong symptom-based shift detected. Baseline was kept close to Layer 1.")

            st.markdown("---")
            st.subheader("History summary")
            st.write(f"Period starts: {[d.strftime('%Y-%m-%d') for d in starts]}")
            st.write(f"Cycle lengths used: {cycle_lengths}")

            history_df = pd.DataFrame({
                "Period Start": [d.strftime("%Y-%m-%d") for d in starts[1:]],
                "Cycle Length From Previous": cycle_lengths
            })
            st.dataframe(history_df, use_container_width=True)

st.markdown("---")
st.info(
    "This is a prototype demo flow: Layer 1 gives the baseline next-period date from cycle history, "
    "and Layer 2 shifts it based on symptoms and cervical mucus logged for a chosen day."
)
