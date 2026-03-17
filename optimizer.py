from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path("models")

FEATURES = [
    "mill_speed_pct",
    "ball_filling_pct",
    "feed_rate_tph",
    "solids_pct",
    "bond_work_index",
    "cyclone_pressure_kpa",
]


def _ensure_models():
    needed = [
        MODELS_DIR / "power_model.pkl",
        MODELS_DIR / "p80_model.pkl",
        MODELS_DIR / "throughput_model.pkl",
    ]
    if not all(p.exists() for p in needed):
        import train_model
        train_model.main()


def load_models():
    _ensure_models()
    power_model = joblib.load(MODELS_DIR / "power_model.pkl")
    p80_model = joblib.load(MODELS_DIR / "p80_model.pkl")
    thr_model = joblib.load(MODELS_DIR / "throughput_model.pkl")
    return power_model, p80_model, thr_model


def evaluate_point(speed, fill, feed, solids, bwi, cyclone):
    power_model, p80_model, thr_model = load_models()
    X = pd.DataFrame([{
        "mill_speed_pct": speed,
        "ball_filling_pct": fill,
        "feed_rate_tph": feed,
        "solids_pct": solids,
        "bond_work_index": bwi,
        "cyclone_pressure_kpa": cyclone,
    }])[FEATURES]

    power = float(power_model.predict(X)[0])
    p80 = float(p80_model.predict(X)[0])
    thr = float(thr_model.predict(X)[0])
    sec = power / max(thr, 1e-6)

    return {
        "power_kw": power,
        "p80_um": p80,
        "throughput_tph": thr,
        "SEC_kwh_per_t": sec,
    }


def optimize(bwi, cyclone, target_p80, min_thr):
    power_model, p80_model, thr_model = load_models()

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
                        "cyclone_pressure_kpa": cyclone,
                    })

    df = pd.DataFrame(rows)
    X = df[FEATURES].copy()
    df["power_kw"] = power_model.predict(X)
    df["p80_um"] = p80_model.predict(X)
    df["throughput_tph"] = thr_model.predict(X)
    df["SEC_kwh_per_t"] = df["power_kw"] / df["throughput_tph"].clip(lower=1e-6)

    feasible = df[
        (df["p80_um"] <= target_p80)
        & (df["throughput_tph"] >= min_thr)
        & (df["power_kw"] <= 1500)
    ]

    if feasible.empty:
        # Fallback: return best overall point and label it as fallback instead of failing.
        best = df.sort_values("SEC_kwh_per_t").iloc[0]
        return {
            "speed_pct_critical": int(best["mill_speed_pct"]),
            "ball_filling_pct": int(best["ball_filling_pct"]),
            "feed_rate_tph": round(float(best["feed_rate_tph"]), 2),
            "solids_pct": int(best["solids_pct"]),
            "power_kw": round(float(best["power_kw"]), 2),
            "p80_um": round(float(best["p80_um"]), 2),
            "throughput_tph": round(float(best["throughput_tph"]), 2),
            "SEC_kwh_per_t": round(float(best["SEC_kwh_per_t"]), 3),
            "feasible": False,
        }

    best = feasible.sort_values("SEC_kwh_per_t").iloc[0]
    return {
        "speed_pct_critical": int(best["mill_speed_pct"]),
        "ball_filling_pct": int(best["ball_filling_pct"]),
        "feed_rate_tph": round(float(best["feed_rate_tph"]), 2),
        "solids_pct": int(best["solids_pct"]),
        "power_kw": round(float(best["power_kw"]), 2),
        "p80_um": round(float(best["p80_um"]), 2),
        "throughput_tph": round(float(best["throughput_tph"]), 2),
        "SEC_kwh_per_t": round(float(best["SEC_kwh_per_t"]), 3),
        "feasible": True,
    }


def generate_contour_data(bwi, cyclone, fill, feed):
    power_model, _, thr_model = load_models()
    rows = []
    for speed in np.arange(68, 81, 2):
        for solids in np.arange(62, 77, 2):
            rows.append({
                "mill_speed_pct": speed,
                "ball_filling_pct": fill,
                "feed_rate_tph": feed,
                "solids_pct": solids,
                "bond_work_index": bwi,
                "cyclone_pressure_kpa": cyclone,
            })

    df = pd.DataFrame(rows)
    X = df[FEATURES].copy()
    df["power_kw"] = power_model.predict(X)
    df["throughput_tph"] = thr_model.predict(X)
    df["SEC_kwh_per_t"] = df["power_kw"] / df["throughput_tph"].clip(lower=1e-6)
    return df


def get_feature_importance():
    power_model, _, _ = load_models()
    df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": power_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    return df
