import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = Path("models")
if not (MODELS_DIR / "power_model.pkl").exists():
    import train_model
    train_model.main()

from optimizer import optimize, evaluate_point, generate_contour_data, get_feature_importance

st.set_page_config(page_title="AI-Assisted Ball Mill Energy Optimization", layout="wide")

st.title("AI-Assisted Ball Mill Energy Optimization")
st.markdown("Hybrid AI-based decision-support dashboard for ball mill power and grind-size optimization.")

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

col1, col2 = st.columns([1.1, 1])
with col1:
    st.subheader("Process Flowsheet")
    st.code(
        """
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
        """
    )
with col2:
    st.subheader("Optimization Goal")
    st.latex(r"SEC = \\frac{Power\ (kW)}{Throughput\ (t/h)}")
    st.markdown(
        """
- P80 ≤ target P80
- Throughput ≥ minimum throughput
- Power ≤ motor/load limit
        """
    )

if st.button("Run Optimization", type="primary"):
    with st.spinner("Running AI optimization..."):
        baseline = evaluate_point(
            speed=base_speed,
            fill=base_fill,
            feed=base_feed,
            solids=base_solids,
            bwi=bwi,
            cyclone=cyclone,
        )
        result = optimize(bwi=bwi, cyclone=cyclone, target_p80=target_p80, min_thr=min_thr)

    if not result["feasible"]:
        st.warning("No fully feasible point met all constraints. Showing the best overall low-energy point instead.")
    else:
        st.success("Optimal feasible operating point found.")

    st.subheader("Baseline vs Optimized Performance")
    c1, c2, c3, c4 = st.columns(4)
    energy_saving_pct = 100 * (baseline["SEC_kwh_per_t"] - result["SEC_kwh_per_t"]) / baseline["SEC_kwh_per_t"]

    c1.metric("Power (kW)", f'{result["power_kw"]:.1f}', f'{result["power_kw"] - baseline["power_kw"]:.1f}')
    c2.metric("P80 (µm)", f'{result["p80_um"]:.1f}', f'{result["p80_um"] - baseline["p80_um"]:.1f}')
    c3.metric("Throughput (t/h)", f'{result["throughput_tph"]:.1f}', f'{result["throughput_tph"] - baseline["throughput_tph"]:.1f}')
    c4.metric("Energy Savings (%)", f"{energy_saving_pct:.1f}%")

    comp = pd.DataFrame({
        "Variable": [
            "Mill Speed (% critical)", "Ball Filling (%)", "Feed Rate (t/h)", "Solids (%)",
            "Power (kW)", "P80 (µm)", "Throughput (t/h)", "SEC (kWh/t)"
        ],
        "Baseline": [
            base_speed, base_fill, base_feed, base_solids,
            round(baseline["power_kw"], 2), round(baseline["p80_um"], 2),
            round(baseline["throughput_tph"], 2), round(baseline["SEC_kwh_per_t"], 3)
        ],
        "Optimized": [
            result["speed_pct_critical"], result["ball_filling_pct"], result["feed_rate_tph"], result["solids_pct"],
            result["power_kw"], result["p80_um"], result["throughput_tph"], result["SEC_kwh_per_t"]
        ]
    })
    st.dataframe(comp, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Power vs Mill Speed**")
        speeds = list(range(65, 83, 2))
        power_vals = []
        for s in speeds:
            pred = evaluate_point(
                speed=s,
                fill=result["ball_filling_pct"],
                feed=result["feed_rate_tph"],
                solids=result["solids_pct"],
                bwi=bwi,
                cyclone=cyclone,
            )
            power_vals.append(pred["power_kw"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(speeds, power_vals, marker="o")
        ax.axvline(result["speed_pct_critical"], linestyle="--")
        ax.set_xlabel("Mill Speed (% critical)")
        ax.set_ylabel("Power (kW)")
        ax.set_title("Predicted Power vs Mill Speed")
        st.pyplot(fig)

    with right:
        st.markdown("**Specific Energy Contour Map**")
        contour_df = generate_contour_data(
            bwi=bwi, cyclone=cyclone, fill=result["ball_filling_pct"], feed=result["feed_rate_tph"]
        )
        pivot = contour_df.pivot(index="solids_pct", columns="mill_speed_pct", values="SEC_kwh_per_t")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        c = ax2.contourf(pivot.columns, pivot.index, pivot.values, levels=15)
        fig2.colorbar(c, ax=ax2, label="SEC (kWh/t)")
        ax2.set_xlabel("Mill Speed (% critical)")
        ax2.set_ylabel("Solids (%)")
        ax2.set_title("SEC Contour Map")
        st.pyplot(fig2)

    st.subheader("AI Feature Importance")
    importance_df = get_feature_importance()
    st.bar_chart(importance_df.set_index("Feature"))
