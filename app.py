import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AI-Assisted Ball Mill Energy Optimization", layout="centered")

MODELS_DIR = Path("models")
POWER_MODEL = MODELS_DIR / "power_model.pkl"

# Train models once if they don't exist yet
if not POWER_MODEL.exists():
    import train_model
    train_model.main()

from optimizer import optimize

st.title("AI-Assisted Ball Mill Energy Optimization")

st.sidebar.header("Process Inputs")
bwi = st.sidebar.slider("Bond Work Index (kWh/t)", 10.0, 20.0, 15.0)
cyclone = st.sidebar.slider("Cyclone Pressure (kPa)", 80, 160, 120)
target_p80 = st.sidebar.slider("Target P80 (µm)", 80, 250, 150)
min_thr = st.sidebar.slider("Minimum Throughput (t/h)", 10, 30, 18)

if st.button("Run Optimization"):
    result = optimize(
        bwi=bwi,
        cyclone=cyclone,
        target_p80=target_p80,
        min_thr=min_thr,
    )

    if result:
        st.success("Optimal operating point found")
        st.json(result)
    else:
        st.error("No feasible solution found in the search range.")
