
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from simulator import simulate_ball_mill

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    rows = []
    rng = np.random.default_rng(42)

    for _ in range(3000):
        speed = rng.uniform(65, 82)
        filling = rng.uniform(25, 40)
        feed = rng.uniform(15, 30)
        solids = rng.uniform(60, 78)
        bwi = rng.uniform(12, 18)
        cyclone = rng.uniform(90, 150)

        power, p80, thr = simulate_ball_mill(speed, filling, feed, solids, bwi, cyclone)

        rows.append({
            "mill_speed_pct": speed,
            "ball_filling_pct": filling,
            "feed_rate_tph": feed,
            "solids_pct": solids,
            "bond_work_index": bwi,
            "cyclone_pressure_kpa": cyclone,
            "power_kw": power,
            "p80_um": p80,
            "throughput_tph": thr,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "ball_mill_demo_data.csv", index=False)

    features = [
        "mill_speed_pct",
        "ball_filling_pct",
        "feed_rate_tph",
        "solids_pct",
        "bond_work_index",
        "cyclone_pressure_kpa",
    ]

    X = df[features]

    power_model = RandomForestRegressor(n_estimators=200, random_state=42)
    p80_model = RandomForestRegressor(n_estimators=200, random_state=42)
    thr_model = RandomForestRegressor(n_estimators=200, random_state=42)

    power_model.fit(X, df["power_kw"])
    p80_model.fit(X, df["p80_um"])
    thr_model.fit(X, df["throughput_tph"])

    joblib.dump(power_model, MODELS_DIR / "power_model.pkl")
    joblib.dump(p80_model, MODELS_DIR / "p80_model.pkl")
    joblib.dump(thr_model, MODELS_DIR / "throughput_model.pkl")


if __name__ == "__main__":
    main()
