import numpy as np

from flroad.client import Client
from flroad.mobility import make_track


def make_client(**extra):
    """Build a client with round numbers, easy to compute by hand."""
    args = dict(
        cid=0,
        t_arrive=0.0,
        kph=36.0,
        t_leave=100.0,
        n_data=1000,
        local_data=None,
        cpu_hz=1e9,
        batch=16,
        epochs=2,
        cycles=1e5,
        eff_capa=1e-27,
        snr_db=0.0,
        bw_hz=1e6,
        ptx=2.0,
        lr=1e-3,
        mom=0.9,
        decay=5e-4,
    )
    args.update(extra)
    return Client(**args)


def test_speed_conversion():
    assert make_client().mps == 10.0


def test_computation_time():
    # (1e5 * 1000 * 2) / 1e9 = 0.2 s
    assert abs(make_client().get_cp_time() - 0.2) < 1e-9


def test_computation_energy():
    # 0.2 s * 1e-27 * (1e9)^3 = 0.2 J
    assert abs(make_client().get_cp_energy() - 0.2) < 1e-9


def test_throughput():
    # SNR 0 dB -> ratio 1 -> log2(2) = 1 -> 1e6 bit/s
    assert abs(make_client().get_throughput() - 1e6) < 1e-3


def test_communication_time_and_energy():
    # 1000 * 16 bits / 1e6 bit/s = 0.016 s, energy = 2 W * 0.016 s
    cl = make_client()
    assert abs(cl.get_co_time(1000, 16) - 0.016) < 1e-9
    assert abs(cl.get_co_energy(1000, 16) - 0.032) < 1e-9


def test_can_finish_uses_remaining_time():
    # needs 0.2 + 0.016 = 0.216 s, leaves at t = 100 s
    cl = make_client()
    assert cl.can_finish(0.0, 1000, 16)
    assert cl.can_finish(99.7, 1000, 16)  # 0.3 s left: enough
    assert not cl.can_finish(99.9, 1000, 16)  # 0.1 s left: too late


def test_throughput_falls_with_distance_to_the_station():
    # 600 m road crossed at 10 m/s; station 50 m from the road middle
    track = make_track(np.full(60, 10.0), 1.0, 600, 50.0)
    cl = make_client(track=track, ple=2.0, d_ref=50.0)
    # t = 30 s: vehicle at the middle, d = d_ref, SNR = 0 dB -> 1e6 bit/s
    assert abs(cl.get_throughput(30.0) - 1e6) < 1e-3
    # t = 0 s: d = 304 m, SNR = -15.7 dB -> about 38 kbit/s
    assert cl.get_throughput(0.0) < 1e5


def test_zero_path_loss_exponent_ignores_the_position():
    track = make_track(np.full(60, 10.0), 1.0, 600, 50.0)
    cl = make_client(track=track, ple=0.0, d_ref=50.0)
    assert cl.get_throughput(0.0) == cl.get_throughput(30.0) == 1e6
