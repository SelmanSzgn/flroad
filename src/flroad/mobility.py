from dataclasses import dataclass

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


@dataclass(frozen=True)
class Track:
    """Position of a vehicle along the road, as a function of time."""

    times: np.ndarray  # seconds since arrival
    pos: np.ndarray  # meters from the start of the road
    road_m: float
    offset_m: float  # lateral distance of the base station

    def distance(self, t_rel: float) -> float:
        """Distance (m) to the base station at t_rel seconds after arrival.

        The base station faces the middle of the road. Outside the stay,
        the position is clamped to the ends of the road.
        """
        x = float(np.interp(t_rel, self.times, self.pos))
        return float(np.hypot(x - self.road_m / 2, self.offset_m))


def make_track(
    speeds: np.ndarray, dt: float, road_m: float, offset_m: float
) -> Track:
    """Build the piecewise-linear trajectory from the speed profile."""
    n = len(speeds)
    # position at the start of each step
    start = np.concatenate(([0.0], np.cumsum(speeds[:-1]) * dt))
    times = np.append(np.arange(n) * dt, stay_time(speeds, dt, road_m))
    pos = np.append(start, road_m)
    return Track(times, pos, road_m, offset_m)
