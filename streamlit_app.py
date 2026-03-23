import streamlit as st
import sklearn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.title("InBalance debug")

st.write("App started")
st.write("sklearn version:", sklearn.__version__)
st.write("pandas version:", pd.__version__)

model_path = Path("layer2a_clf_mucus.joblib")
features_path = Path("layer2a_feature_cols.joblib")

st.write("Model exists:", model_path.exists())
st.write("Features exists:", features_path.exists())

if model_path.exists():
    st.write("Trying to load model...")
    try:
        clf = joblib.load(model_path)
        st.success("Model loaded")
        st.write(type(clf))
    except Exception as e:
        st.error(f"Model load failed: {e}")

if features_path.exists():
    st.write("Trying to load feature cols...")
    try:
        feature_cols = joblib.load(features_path)
        st.success("Feature cols loaded")
        st.write("Number of feature cols:", len(feature_cols))
        st.write(feature_cols[:10] if len(feature_cols) > 10 else feature_cols)
    except Exception as e:
        st.error(f"Feature load failed: {e}")
