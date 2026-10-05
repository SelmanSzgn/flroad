import numpy as np


def speed_profile(
    v0: float,
    vmin: float,
    vmax: float,
    alpha: float,
    sigma: float,
    road_m: float,
    dt: float,
) -> np.ndarray:
    """Gauss-Markov speeds (m/s), one per step, until the road is covered.

    v0 is the cruising speed the speed reverts to. With alpha = 1 the
    speed is constant and no random number is drawn.
    """
    speeds: list[float] = []
    v, covered = v0, 0.0
    while covered < road_m:
        speeds.append(v)
        covered += v * dt
        noise = float(np.random.normal()) if alpha < 1 else 0.0
        v = alpha * v + (1 - alpha) * v0 + np.sqrt(1 - alpha**2) * sigma * noise
        v = min(max(v, vmin), vmax)
    return np.array(speeds)


def stay_time(speeds: np.ndarray, dt: float, road_m: float) -> float:
    """Time (s) needed to cover road_m, interpolating in the last step."""
    before_last = float(speeds[:-1].sum() * dt)
    return (len(speeds) - 1) * dt + (road_m - before_last) / float(speeds[-1])
