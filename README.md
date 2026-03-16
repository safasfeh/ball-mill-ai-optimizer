# AI-Assisted Ball Mill Energy Optimizer (Demo)

A runnable Streamlit application that:

1. Generates a synthetic demo dataset for a ball-mill grinding circuit.
2. Trains machine-learning surrogate models for mill power, product P80, and throughput.
3. Optimizes controllable operating variables to minimize specific energy (kWh/t), while satisfying grind-size and throughput constraints.
4. Visualizes baseline vs optimized performance, feasible operating windows, and feature importance.

## Files

- `app.py` — Streamlit UI
- `train_model.py` — synthetic data generation + model training
- `optimizer.py` — constrained grid-search optimizer
- `simulator.py` — synthetic process simulator used to build the demo dataset
- `utils/plots.py` — plotting helpers
- `data/` — generated CSV dataset
- `models/` — trained models and metadata

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- The dataset is synthetic and intended for demonstration and teaching only.
- The process equations are simplified but intentionally grounded in mineral-processing logic: higher ore hardness and finer target grind increase energy demand; speed, filling, solids, and cyclone pressure influence power, size reduction, and throughput.
- The default ML model is `RandomForestRegressor` from scikit-learn for portability and robustness.
