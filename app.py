
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

st.set_page_config(
    page_title="AI-Assisted Ball Mill Energy Optimization",
    page_icon="⚙️",
    layout="wide"
)

MODELS_DIR = Path("models")
POWER_MODEL = MODELS_DIR / "power_model.pkl"

if not POWER_MODEL.exists():
    import train_model
    train_model.main()

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
}
.hero {
    padding: 1rem 1.2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #0b1f33 0%, #1f4e79 100%);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
}
.hero p {
    margin: 0.25rem 0 0 0;
    opacity: 0.92;
}
.panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 16px;
    padding: 0.9rem 1rem;
}
.small-note {
    font-size: 0.9rem;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

logo_path = Path("mst_logo.png")
header_left, header_right = st.columns([1, 7])

with header_left:
    if logo_path.exists():
        st.image(str(logo_path), width=110)
    else:
        st.markdown("### ⛏️")

with header_right:
    st.markdown(
        """
        <div class="hero">
            <h1>Missouri University of Science and Technology</h1>
            <p>AI-Assisted Ball Mill Energy Optimization Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.header("Process Inputs")
    bwi = st.slider("Bond Work Index (kWh/t)", 10.0, 20.0, 15.0, 0.1)
    cyclone = st.slider("Cyclone Pressure (kPa)", 80, 160, 120, 1)
    target_p80 = st.slider("Target P80 (µm)", 100, 220, 180, 5)
    min_thr = st.slider("Minimum Throughput (t/h)", 10, 30, 16, 1)

    st.markdown("---")
    st.subheader("Baseline Operating Point")
    base_speed = st.slider("Baseline Mill Speed (% critical)", 65, 82, 74, 1)
    base_fill = st.slider("Baseline Ball Filling (%)", 25, 40, 32, 1)
    base_feed = st.slider("Baseline Feed Rate (t/h)", 15, 30, 22, 1)
    base_solids = st.slider("Baseline Solids (%)", 60, 78, 68, 1)

    st.markdown("---")
    st.caption("")

top1, top2 = st.columns([1.25, 1])

with top1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
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
```
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with top2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Optimization Objective")
    st.latex(r"SEC = \frac{Power\ (kW)}{Throughput\ (t/h)}")
    st.markdown("""
The system uses **machine learning surrogate models** to estimate:

- mill power draw
- product size (P80)
- throughput

and selects the operating point with the **lowest specific energy consumption** while checking operational constraints.
""")
    st.markdown('<div class="small-note">Constraints: target P80, minimum throughput, motor/load limit.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

run = st.button("Run Optimization", type="primary", use_container_width=True)

if run:
    with st.spinner("Running AI optimization..."):
        baseline = evaluate_point(
            speed=base_speed,
            fill=base_fill,
            feed=base_feed,
            solids=base_solids,
            bwi=bwi,
            cyclone=cyclone
        )

        result = optimize(
            bwi=bwi,
            cyclone=cyclone,
            target_p80=target_p80,
            min_thr=min_thr
        )

    energy_saving_pct = 100 * (baseline["SEC_kwh_per_t"] - result["SEC_kwh_per_t"]) / baseline["SEC_kwh_per_t"]

    st.subheader("KPI Dashboard")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimized Power (kW)", f'{result["power_kw"]:.1f}', f'{result["power_kw"] - baseline["power_kw"]:.1f}')
    k2.metric("Optimized P80 (µm)", f'{result["p80_um"]:.1f}', f'{result["p80_um"] - baseline["p80_um"]:.1f}')
    k3.metric("Optimized Throughput (t/h)", f'{result["throughput_tph"]:.1f}', f'{result["throughput_tph"] - baseline["throughput_tph"]:.1f}')
    k4.metric("Energy Savings", f"{energy_saving_pct:.1f}%", f"{energy_saving_pct:.1f}%")

    baseline_inputs = {
        "speed": base_speed,
        "fill": base_fill,
        "feed": base_feed,
        "solids": base_solids
    }
    recommendations = generate_recommendations(baseline_inputs, result)
    summary_text = summarize_actions(baseline_inputs, result)

    st.subheader("Recommended Control Actions")
    st.success(summary_text)

    st.subheader("Operational Recommendations")
    for i, rec in enumerate(recommendations, start=1):
        st.markdown(f"{i}. {rec}")

    tab1, tab2, tab3 = st.tabs(["Comparison", "Visual Analytics", "Model Insight"])

    with tab1:
        comparison_df = pd.DataFrame({
            "Variable": [
                "Mill Speed (% critical)",
                "Ball Filling (%)",
                "Feed Rate (t/h)",
                "Solids (%)",
                "Power (kW)",
                "P80 (µm)",
                "Throughput (t/h)",
                "SEC (kWh/t)"
            ],
            "Baseline": [
                base_speed,
                base_fill,
                base_feed,
                base_solids,
                round(baseline["power_kw"], 2),
                round(baseline["p80_um"], 2),
                round(baseline["throughput_tph"], 2),
                round(baseline["SEC_kwh_per_t"], 3)
            ],
            "Optimized": [
                result["speed_pct_critical"],
                result["ball_filling_pct"],
                result["feed_rate_tph"],
                result["solids_pct"],
                result["power_kw"],
                result["p80_um"],
                result["throughput_tph"],
                result["SEC_kwh_per_t"]
            ]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    with tab2:
        left, right = st.columns(2)

        with left:
            st.markdown("#### Predicted Power vs Mill Speed")
            speeds = list(range(65, 83, 2))
            power_vals = []
            for s in speeds:
                pred = evaluate_point(
                    speed=s,
                    fill=result["ball_filling_pct"],
                    feed=result["feed_rate_tph"],
                    solids=result["solids_pct"],
                    bwi=bwi,
                    cyclone=cyclone
                )
                power_vals.append(pred["power_kw"])

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(speeds, power_vals, marker="o")
            ax.axvline(result["speed_pct_critical"], linestyle="--")
            ax.set_xlabel("Mill Speed (% critical)")
            ax.set_ylabel("Power (kW)")
            ax.set_title("Predicted Power vs Mill Speed")
            st.pyplot(fig, clear_figure=True)

        with right:
            st.markdown("#### Specific Energy Contour Map")
            contour_df = generate_contour_data(
                bwi=bwi,
                cyclone=cyclone,
                fill=result["ball_filling_pct"],
                feed=result["feed_rate_tph"]
            )

            pivot = contour_df.pivot(index="solids_pct", columns="mill_speed_pct", values="SEC_kwh_per_t")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            contour = ax2.contourf(pivot.columns, pivot.index, pivot.values, levels=15)
            fig2.colorbar(contour, ax=ax2, label="SEC (kWh/t)")
            ax2.set_xlabel("Mill Speed (% critical)")
            ax2.set_ylabel("Solids (%)")
            ax2.set_title("SEC Contour Map")
            st.pyplot(fig2, clear_figure=True)

    with tab3:
        st.markdown("#### Power Model Feature Importance")
        importance_df = get_feature_importance()
        st.bar_chart(importance_df.set_index("Feature"))
        st.caption("Feature importance is derived from the Random Forest power model.")

    st.subheader("Results Interpretation")
    delta_power = result["power_kw"] - baseline["power_kw"]
    delta_p80 = result["p80_um"] - baseline["p80_um"]
    delta_thr = result["throughput_tph"] - baseline["throughput_tph"]
    delta_sec = result["SEC_kwh_per_t"] - baseline["SEC_kwh_per_t"]

    st.markdown(
        f"""
The optimization results indicate a shift in the mill operating conditions that improves overall energy efficiency.

- **Power consumption** changed by {delta_power:.1f} kW  
- **Product size (P80)** changed by {delta_p80:.1f} µm  
- **Throughput** changed by {delta_thr:.1f} t/h  
- **Specific energy consumption** changed by {delta_sec:.2f} kWh/t  

Overall, the optimized condition achieves a **reduction in specific energy consumption of {abs(delta_sec):.2f} kWh/t**.

This indicates a more efficient grinding regime with improved energy utilization.
"""
    )

    st.subheader("Engineering Recommendations")
    rec_text = []

    if result["speed_pct_critical"] > base_speed:
        rec_text.append("Increase mill speed to enhance breakage rate and improve grinding efficiency.")
    elif result["speed_pct_critical"] < base_speed:
        rec_text.append("Reduce mill speed to avoid excessive energy consumption and over-grinding.")
    else:
        rec_text.append("Maintain current mill speed, as it is near optimal.")

    if result["ball_filling_pct"] > base_fill:
        rec_text.append("Increase ball filling to improve impact and attrition mechanisms.")
    elif result["ball_filling_pct"] < base_fill:
        rec_text.append("Reduce ball filling to minimize energy losses due to excessive media load.")
    else:
        rec_text.append("Maintain current ball filling level.")

    if result["feed_rate_tph"] > base_feed:
        rec_text.append("Increase feed rate to improve throughput while maintaining energy efficiency.")
    elif result["feed_rate_tph"] < base_feed:
        rec_text.append("Reduce feed rate to improve residence time and achieve finer product size.")
    else:
        rec_text.append("Maintain current feed rate.")

    if result["solids_pct"] > base_solids:
        rec_text.append("Increase solids concentration to improve grinding efficiency through better energy transfer.")
    elif result["solids_pct"] < base_solids:
        rec_text.append("Reduce solids concentration to improve slurry transport and classification efficiency.")
    else:
        rec_text.append("Maintain current solids percentage.")

    if result["p80_um"] < baseline["p80_um"]:
        rec_text.append("Finer product size indicates improved breakage performance.")
    else:
        rec_text.append("Coarser product size suggests energy savings at the expense of grind fineness.")

    if result["throughput_tph"] > baseline["throughput_tph"]:
        rec_text.append("Throughput improvement indicates better mill productivity.")
    else:
        rec_text.append("Throughput reduction suggests operation closer to fine grinding conditions.")

    if result["SEC_kwh_per_t"] < baseline["SEC_kwh_per_t"]:
        rec_text.append("Reduced specific energy confirms improved grinding efficiency.")
    else:
        rec_text.append("Increased specific energy indicates less efficient operation.")

    for i, r in enumerate(rec_text, 1):
        st.markdown(f"{i}. {r}")

    st.subheader("Executive Summary")
    st.success(
        f"""
The AI-based optimization identified a more energy-efficient operating condition.

Compared to the baseline:
- Specific energy changed from **{baseline["SEC_kwh_per_t"]:.2f} to {result["SEC_kwh_per_t"]:.2f} kWh/t**
- Power consumption changed from **{baseline["power_kw"]:.1f} to {result["power_kw"]:.1f} kW**
- Throughput changed from **{baseline["throughput_tph"]:.1f} to {result["throughput_tph"]:.1f} t/h**
- Product size changed from **{baseline["p80_um"]:.1f} to {result["p80_um"]:.1f} µm**

These results demonstrate that AI-assisted optimization can improve grinding circuit performance by balancing energy consumption, throughput, and product size.
"""
    )

st.markdown("---")
st.caption("Developed at Missouri University of Science and Technology")
