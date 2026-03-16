from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path("models")

def load_models():
    power_path = MODELS_DIR / "power_model.pkl"
    p80_path = MODELS_DIR / "p80_model.pkl"
    thr_path = MODELS_DIR / "throughput_model.pkl"

    if not (power_path.exists() and p80_path.exists() and thr_path.exists()):
        import train_model
        train_model.main()

    power_model = joblib.load(power_path)
    p80_model = joblib.load(p80_path)
    thr_model = joblib.load(thr_path)
    return power_model, p80_model, thr_model


def optimize(bwi, cyclone, target_p80, min_thr):
    power_model, p80_model, thr_model = load_models()

    best = None

    for speed in np.arange(65, 83, 1):
        for fill in np.arange(25, 41, 1):
            for feed in np.arange(15, 31, 1):
                for solids in np.arange(60, 79, 2):

                    row = pd.DataFrame([{
                        "mill_speed_pct": speed,
                        "ball_filling_pct": fill,
                        "feed_rate_tph": feed,
                        "solids_pct": solids,
                        "bond_work_index": bwi,
                        "cyclone_pressure_kpa": cyclone
                    }])

                    power = float(power_model.predict(row)[0])
                    p80 = float(p80_model.predict(row)[0])
                    thr = float(thr_model.predict(row)[0])

                    sec = power / thr

                    if p80 > target_p80:
                        continue
                    if thr < min_thr:
                        continue
                    if power > 1500:
                        continue

                    if best is None or sec < best["SEC"]:
                        best = {
                            "speed_pct_critical": speed,
                            "ball_filling_pct": fill,
                            "feed_rate_tph": feed,
                            "solids_pct": solids,
                            "power_kw": round(power, 2),
                            "p80_um": round(p80, 2),
                            "throughput_tph": round(thr, 2),
                            "SEC_kwh_per_t": round(sec, 3),
                        }

    return best
