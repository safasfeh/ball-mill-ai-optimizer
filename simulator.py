import numpy as np

def simulate_ball_mill(speed, filling, feed_rate, solids, bond_work_index, cyclone_pressure):
    power = (
        500
        + 8 * speed
        + 5 * filling
        + 6 * feed_rate
        + 2 * solids
        + 10 * bond_work_index
        + 1.5 * cyclone_pressure
        + np.random.normal(0, 15)
    )

    p80 = (
        250
        - 0.8 * speed
        - 0.5 * filling
        - 0.4 * solids
        + 0.3 * feed_rate
        + 1.2 * bond_work_index
        + np.random.normal(0, 5)
    )

    throughput = feed_rate * (1 + (speed - 70) / 200)

    return power, p80, throughput
