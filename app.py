import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from optimizer import (
    optimize,
    evaluate_point,
    generate_contour_data,
    get_feature_importance,
    generate_recommendations,
    summarize_actions,
)

st.set_page_config(page_title="AI-Assisted Ball Mill Energy Optimization", layout="wide")

st.title("AI-Assisted Ball Mill Energy Optimization")

bwi = st.slider("Bond Work Index", 10.0, 20.0, 15.0)
cyclone = st.slider("Cyclone Pressure", 80, 160, 120)
target_p80 = st.slider("Target P80", 100, 220, 180)
min_thr = st.slider("Min Throughput", 10, 30, 16)

base_speed = st.slider("Baseline Speed", 65, 82, 74)
base_fill = st.slider("Baseline Filling", 25, 40, 32)
base_feed = st.slider("Baseline Feed", 15, 30, 22)
base_solids = st.slider("Baseline Solids", 60, 78, 68)

if st.button("Run Optimization"):
    baseline = evaluate_point(base_speed, base_fill, base_feed, base_solids, bwi, cyclone)
    result = optimize(bwi, cyclone, target_p80, min_thr)

    st.subheader("Executive Summary")
    st.write(f"SEC: {baseline['SEC_kwh_per_t']:.2f} → {result['SEC_kwh_per_t']:.2f}")

    st.subheader("Recommendations")
    recs = generate_recommendations(
        {"speed":base_speed,"fill":base_fill,"feed":base_feed,"solids":base_solids},
        result
    )
    for r in recs:
        st.write("- " + r)
