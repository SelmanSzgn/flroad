import numpy as np

from flroad.mobility import make_track, speed_profile, stay_time
from flroad.utils import set_seed


def test_alpha_one_keeps_a_constant_speed():
    s = speed_profile(10.0, 5.0, 20.0, 1.0, 2.0, 600, 1.0)
    assert np.all(s == 10.0)
    assert abs(stay_time(s, 1.0, 600) - 60.0) < 1e-9


def test_stay_time_interpolates_in_the_last_step():
    # 10 m + 10 m + 20 m would reach 40 m; the road is 35 m long:
    # 2 full steps (20 m), then 15 m at 20 m/s = 0.75 s
    s = np.array([10.0, 10.0, 20.0])
    assert abs(stay_time(s, 1.0, 35) - 2.75) < 1e-9


def test_speeds_stay_within_bounds_and_vary():
    set_seed(0)
    s = speed_profile(10.0, 5.0, 20.0, 0.5, 5.0, 600, 1.0)
    assert s.min() >= 5.0 and s.max() <= 20.0
    assert s.std() > 0


def test_stay_time_is_between_the_extreme_speeds():
    set_seed(1)
    s = speed_profile(10.0, 5.0, 20.0, 0.9, 5.0, 600, 1.0)
    assert 600 / 20.0 <= stay_time(s, 1.0, 600) <= 600 / 5.0


def test_profile_is_reproducible():
    set_seed(3)
    a = speed_profile(10.0, 5.0, 20.0, 0.8, 3.0, 600, 1.0)
    set_seed(3)
    b = speed_profile(10.0, 5.0, 20.0, 0.8, 3.0, 600, 1.0)
    assert np.array_equal(a, b)


def test_track_position_and_distance():
    # positions at t = 0, 1, 2, 2.75 s are 0, 10, 20, 35 m (road: 35 m)
    tr = make_track(np.array([10.0, 10.0, 20.0]), 1.0, 35, 40.0)
    edge = np.hypot(17.5, 40.0)  # road middle is at 17.5 m
    assert abs(tr.distance(0.0) - edge) < 1e-9
    assert abs(tr.distance(2.75) - edge) < 1e-9
    assert abs(tr.distance(1.5) - np.hypot(2.5, 40.0)) < 1e-9
