from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path("models")

def load_models():
    power_model = joblib.load(MODELS_DIR / "power_model.pkl")
    p80_model = joblib.load(MODELS_DIR / "p80_model.pkl")
    thr_model = joblib.load(MODELS_DIR / "throughput_model.pkl")
    return power_model, p80_model, thr_model


def optimize(bwi, cyclone, target_p80, min_thr):
    power_model, p80_model, thr_model = load_models()

    best = None

    # Reduced search ranges for faster cloud performance
    speed_range = np.arange(68, 81, 2)
    fill_range = np.arange(28, 39, 2)
    feed_range = np.arange(18, 29, 2)
    solids_range = np.arange(62, 77, 3)

    rows = []
    for speed in speed_range:
        for fill in fill_range:
            for feed in feed_range:
                for solids in solids_range:
                rows.append({
                    "mill_speed_pct": speed,
                    "ball_filling_pct": fill,
                    "feed_rate_tph": feed,
                    "solids_pct": solids,
                    "bond_work_index": bwi,
                    "cyclone_pressure_kpa": cyclone
                })

    df = pd.DataFrame(rows)

    power_pred = power_model.predict(df)
    p80_pred = p80_model.predict(df)
    thr_pred = thr_model.predict(df)

    df["power_kw"] = power_pred
    df["p80_um"] = p80_pred
    df["throughput_tph"] = thr_pred
    df["SEC_kwh_per_t"] = df["power_kw"] / df["throughput_tph"]

    df = df[
        (df["p80_um"] <= target_p80) &
        (df["throughput_tph"] >= min_thr) &
        (df["power_kw"] <= 1500)
    ]

    if df.empty:
        return None

    best_row = df.sort_values("SEC_kwh_per_t").iloc[0]

    return {
        "speed_pct_critical": int(best_row["mill_speed_pct"]),
        "ball_filling_pct": int(best_row["ball_filling_pct"]),
        "feed_rate_tph": round(float(best_row["feed_rate_tph"]), 2),
        "solids_pct": int(best_row["solids_pct"]),
        "power_kw": round(float(best_row["power_kw"]), 2),
        "p80_um": round(float(best_row["p80_um"]), 2),
        "throughput_tph": round(float(best_row["throughput_tph"]), 2),
        "SEC_kwh_per_t": round(float(best_row["SEC_kwh_per_t"]), 3),
    }
