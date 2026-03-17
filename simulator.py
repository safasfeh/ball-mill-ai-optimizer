
import numpy as np


def simulate_ball_mill(speed, filling, feed_rate, solids, bond_work_index, cyclone_pressure):
    power = (
        500
        + 7.5 * speed
        + 4.5 * filling
        + 5.5 * feed_rate
        + 1.8 * solids
        + 11.0 * bond_work_index
        + 1.2 * cyclone_pressure
        + np.random.normal(0, 12)
    )

    p80 = (
        260
        - 1.0 * speed
        - 0.6 * filling
        - 0.45 * solids
        + 0.2 * feed_rate
        + 1.0 * bond_work_index
        + np.random.normal(0, 4)
    )

    throughput = feed_rate * (1 + (speed - 70) / 180) + np.random.normal(0, 0.3)

    return power, p80, throughput
