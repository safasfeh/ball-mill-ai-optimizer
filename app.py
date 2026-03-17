import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from optimizer import optimize, evaluate_point, generate_contour_data, get_feature_importance

st.set_page_config(
    page_title="AI-Assisted Ball Mill Energy Optimization",
    layout="wide"
)

st.title("AI-Assisted Ball Mill Energy Optimization")
st.markdown("Hybrid AI-based decision-support dashboard for ball mill power and grind-size optimization.")

MODELS_DIR = Path("models")
POWER_MODEL = MODELS_DIR / "power_model.pkl"

if not POWER_MODEL.exists():
    import train_model
    train_model.main()

# ---------------- Sidebar ----------------
st.sidebar.header("Process Inputs")

bwi = st.sidebar.slider("Bond Work Index (kWh/t)", 10.0, 20.0, 15.0, 0.1)
cyclone = st.sidebar.slider("Cyclone Pressure (kPa)", 80, 160, 120, 1)
target_p80 = st.sidebar.slider("Target P80 (µm)", 100, 220, 180, 5)
min_thr = st.sidebar.slider("Minimum Throughput (t/h)", 10, 30, 16, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Baseline Operating Point")
base_speed = st.sidebar.slider("Baseline Mill Speed (% critical)", 65, 82, 74, 1)
base_fill = st.sidebar.slider("Baseline Ball Filling (%)", 25, 40, 32, 1)
base_feed = st.sidebar.slider("Baseline Feed Rate (t/h)", 15, 30, 22, 1)
base_solids = st.sidebar.slider("Baseline Solids (%)", 60, 78, 68, 1)

# ---------------- Layout ----------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Process Flowsheet")
    st.markdown(
        """
```text
        Ore Feed
           ↓
      ┌───────────┐
      │ Ball Mill │
      └───────────┘
           ↓
      ┌───────────┐
      │ Cyclone   │
      └───────────┘
       ↙         ↘
 Overflow      Underflow
(Product)   (Circulating Load)
                ↓
             Ball Mill
