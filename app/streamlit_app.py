import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from engine.config import SUPPORTED_SYMPTOMS, MUCUS_OPTIONS
from engine.layer_fusion import get_fused_output
from engine.recommender import get_recommendations

st.set_page_config(page_title="InBalance Cycle Engine", layout="centered")

st.title("InBalance Cycle Engine")
st.write("Layer 1 = history, Layer 2 = trained symptom model, Fusion = final output")

st.subheader("Period history")
period_text = st.text_area(
    "Enter period start dates, one per line (YYYY-MM-DD)",
    value="2026-01-01\n2026-01-29\n2026-02-26"
)

st.subheader("Symptoms")
selected_symptoms = st.multiselect("Select symptoms", SUPPORTED_SYMPTOMS)

st.subheader("Additional inputs")
appetite = st.selectbox("Appetite change", [0, 1], help="0 = no, 1 = yes")
exerciselevel = st.selectbox("Exercise level change", [0, 1], help="0 = no, 1 = yes")

st.subheader("Cervical mucus")
cervical_mucus = st.selectbox("Cervical mucus", MUCUS_OPTIONS, index=MUCUS_OPTIONS.index("unknown"))

if st.button("Run prediction"):
    period_starts = [x.strip() for x in period_text.splitlines() if x.strip()]

    result = get_fused_output(
        period_starts=period_starts,
        symptoms=selected_symptoms,
        cervical_mucus=cervical_mucus,
        appetite=appetite,
        exerciselevel=exerciselevel,
    )

    st.subheader("Final Output")
    st.json({
        "mode": result["mode"],
        "final_phase": result["final_phase"],
        "final_phase_probs": result["final_phase_probs"],
    })

    st.subheader("Layer 1 Output")
    st.json(result["layer1"])

    if result["layer2"] is not None:
        st.subheader("Layer 2 Output")
        st.json({
            "top_phase": result["layer2"]["top_phase"],
            "phase_probs": result["layer2"]["phase_probs"],
        })

    st.subheader("Recommendations")
    st.json(get_recommendations(result["final_phase"]))
