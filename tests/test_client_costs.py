import sys
from pathlib import Path

# Make the flat modules (client.py, ...) importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import Client


def make_client():
    """Build a client with round numbers, easy to compute by hand."""
    return Client(
        cid=0, t_arrive=0.0, kph=36.0, t_leave=100.0, n_data=1000,
        local_data=None, cpu_hz=1e9, batch=16, epochs=2, cycles=1e5,
        eff_capa=1e-27, snr_db=0.0, bw_hz=1e6, ptx=2.0, lr=1e-3,
        mom=0.9, decay=5e-4,
    )


def test_speed_conversion():
    # 36 km/h = 10 m/s
    assert make_client().mps == 10.0


def test_computation_time():
    # (1e5 cycles * 1000 data * 2 epochs) / 1e9 Hz = 0.2 s
    assert abs(make_client().get_cp_time() - 0.2) < 1e-9


def test_throughput():
    # SNR 0 dB = ratio 1, so log2(1 + 1) = 1, throughput = 1e6 bit/s
    assert abs(make_client().get_throughput() - 1e6) < 1e-3


def test_communication_time():
    # 1000 params * 16 bits = 16000 bits, at 1e6 bit/s = 0.016 s
    assert abs(make_client().get_co_time(1000, 16) - 0.016) < 1e-9
