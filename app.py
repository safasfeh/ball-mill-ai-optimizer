from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from optimizer import load_models, optimize_ball_mill, predict_point
from train_model import DATASET_PATH, FEATURES, METADATA_PATH, generate_demo_dataset, train_and_save_models
from utils.plots import plot_baseline_vs_optimized, plot_feature_importance, plot_feasible_map

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Ball Mill AI Optimizer", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


@st.cache_resource(show_spinner=False)
def load_models_cached():
    return load_models()


def models_ready() -> bool:
    return METADATA_PATH.exists() and all((MODELS_DIR / f"{t}_model.pkl").exists() for t in ["power_kw", "p80_um", "throughput_tph"])


st.title("AI-Assisted Ball Mill Specific Energy Optimizer")
st.caption("Synthetic-data demonstration app for training, prediction, and constrained optimization.")

with st.sidebar:
    st.header("1) Dataset and model setup")
    n_samples = st.number_input("Synthetic dataset size", min_value=500, max_value=10000, value=2500, step=250)
    random_state = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)

    if st.button("Generate dataset + train models", use_container_width=True):
        with st.spinner("Generating synthetic dataset and training models..."):
            df = generate_demo_dataset(n_samples=int(n_samples), random_state=int(random_state))
            metrics = train_and_save_models(df, random_state=int(random_state))
            st.cache_data.clear()
            st.cache_resource.clear()
        st.success("Training complete.")
        st.json(metrics)

if not models_ready():
    st.warning("No trained models were found. Use the sidebar button to generate the demo dataset and train the models.")
    st.stop()

models, feature_names = load_models_cached()
metadata = json.loads(METADATA_PATH.read_text())

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("2) Fixed ore and equipment inputs")
    feed_f80_um = st.slider("Feed F80 (µm)", 800, 2600, 1600, 50)
    bond_work_index = st.slider("Bond work index (kWh/t)", 10.0, 19.0, 14.5, 0.1)
    cyclone_pressure_kpa = st.slider("Cyclone pressure (kPa)", 80, 160, 120, 1)
    mill_diameter_m = st.slider("Mill diameter (m)", 2.7, 4.5, 3.6, 0.1)
    mill_length_m = st.slider("Mill length (m)", 3.2, 6.0, 4.8, 0.1)
    liner_factor = st.slider("Liner factor", 0.85, 1.10, 1.00, 0.01)

with col2:
    st.subheader("3) Baseline operating point")
    base_speed = st.slider("Baseline mill speed (% critical)", 65, 82, 74, 1)
    base_filling = st.slider("Baseline ball filling (%)", 24, 40, 32, 1)
    base_feed = st.slider("Baseline feed rate (t/h)", 12, 40, 24, 1)
    base_solids = st.slider("Baseline solids (%)", 60, 78, 69, 1)

st.subheader("4) Optimization constraints")
col3, col4, col5 = st.columns(3)
with col3:
    target_p80_um = st.number_input("Target P80 upper limit (µm)", min_value=60.0, max_value=300.0, value=150.0, step=5.0)
with col4:
    min_throughput_tph = st.number_input("Minimum throughput (t/h)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
with col5:
    motor_limit_kw = st.number_input("Motor/load limit (kW)", min_value=200.0, max_value=2000.0, value=1400.0, step=50.0)

st.subheader("5) Search ranges for optimization")
col6, col7, col8, col9 = st.columns(4)
with col6:
    speed_range = st.slider("Speed range", 65, 82, (68, 80), 1)
with col7:
    filling_range = st.slider("Filling range", 24, 40, (26, 38), 1)
with col8:
    feed_range = st.slider("Feed rate range", 12, 40, (18, 32), 1)
with col9:
    solids_range = st.slider("Solids range", 60, 78, (64, 75), 1)

fixed_inputs = {
    "bond_work_index": float(bond_work_index),
    "cyclone_pressure_kpa": float(cyclone_pressure_kpa),
    "feed_f80_um": float(feed_f80_um),
    "mill_diameter_m": float(mill_diameter_m),
    "mill_length_m": float(mill_length_m),
    "liner_factor": float(liner_factor),
}

baseline_row = {
    **fixed_inputs,
    "mill_speed_pct": float(base_speed),
    "ball_filling_pct": float(base_filling),
    "feed_rate_tph": float(base_feed),
    "solids_pct": float(base_solids),
}
baseline = predict_point(baseline_row, models, feature_names)

run = st.button("Run optimization", type="primary", use_container_width=True)

if run:
    bounds = {
        "mill_speed_pct": (float(speed_range[0]), float(speed_range[1]), 1.0),
        "ball_filling_pct": (float(filling_range[0]), float(filling_range[1]), 1.0),
        "feed_rate_tph": (float(feed_range[0]), float(feed_range[1]), 1.0),
        "solids_pct": (float(solids_range[0]), float(solids_range[1]), 1.0),
    }

    with st.spinner("Evaluating candidate operating points..."):
        best, results_df = optimize_ball_mill(
            fixed_inputs=fixed_inputs,
            bounds=bounds,
            target_p80_um=float(target_p80_um),
            min_throughput_tph=float(min_throughput_tph),
            motor_limit_kw=float(motor_limit_kw),
        )

    st.subheader("6) Results")
    result_cols = st.columns(4)
    result_cols[0].metric("Baseline SEC (kWh/t)", f"{baseline['specific_energy_kwhpt']:.2f}")
    result_cols[1].metric("Baseline power (kW)", f"{baseline['power_kw']:.1f}")
    result_cols[2].metric("Baseline P80 (µm)", f"{baseline['p80_um']:.1f}")
    result_cols[3].metric("Baseline throughput (t/h)", f"{baseline['throughput_tph']:.1f}")

    if best is None:
        st.error("No feasible operating point was found within the selected search ranges.")
    else:
        improvement = (baseline["specific_energy_kwhpt"] - best["specific_energy_kwhpt"]) / baseline["specific_energy_kwhpt"] * 100.0
        opt_cols = st.columns(4)
        opt_cols[0].metric("Optimized SEC (kWh/t)", f"{best['specific_energy_kwhpt']:.2f}", delta=f"{-improvement:.1f}%")
        opt_cols[1].metric("Optimized power (kW)", f"{best['power_kw']:.1f}")
        opt_cols[2].metric("Optimized P80 (µm)", f"{best['p80_um']:.1f}")
        opt_cols[3].metric("Optimized throughput (t/h)", f"{best['throughput_tph']:.1f}")

        st.write("**Recommended operating point**")
        st.json({k: round(v, 3) if isinstance(v, float) else v for k, v in best.items() if k not in ["feasible"]})

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(plot_baseline_vs_optimized(baseline, best), clear_figure=True)
        with chart_col2:
            st.pyplot(plot_feasible_map(results_df, "mill_speed_pct", "solids_pct"), clear_figure=True)

        st.write("**Top feasible candidates**")
        display_cols = [
            "mill_speed_pct",
            "ball_filling_pct",
            "feed_rate_tph",
            "solids_pct",
            "power_kw",
            "p80_um",
            "throughput_tph",
            "specific_energy_kwhpt",
        ]
        st.dataframe(results_df[results_df["feasible"]][display_cols].head(15), use_container_width=True)

st.subheader("7) Model diagnostics")
diag1, diag2, diag3 = st.columns(3)
for col, target in zip([diag1, diag2, diag3], ["power_kw", "p80_um", "throughput_tph"]):
    with col:
        st.write(f"**{target} model**")
        st.write(f"R²: {metadata['metrics'][target]['r2']:.3f}")
        st.write(f"MAE: {metadata['metrics'][target]['mae']:.3f}")
        st.pyplot(plot_feature_importance(models[target], feature_names), clear_figure=True)

st.subheader("8) Dataset preview")
st.dataframe(load_dataset().head(20), use_container_width=True)
