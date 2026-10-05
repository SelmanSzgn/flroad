import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from flroad.config import Config
from flroad.data import sample_local_data
from flroad.mobility import Track, make_track, speed_profile, stay_time


class Client:
    def __init__(
        self,
        cid: int,
        t_arrive: float,
        kph: float,
        t_leave: float,
        n_data: int,
        local_data: Subset,
        cpu_hz: float,
        batch: int,
        epochs: int,
        cycles: float,
        eff_capa: float,
        snr_db: float,
        bw_hz: float,
        ptx: float,
        lr: float,
        mom: float,
        decay: float,
        track: Track | None = None,
        ple: float = 0.0,
        d_ref: float = 1.0,
    ) -> None:
        # client id
        self.cid = cid
        # arrival timestamp
        self.t_arrive = t_arrive
        # cruising speed in kilometer per hour
        self.kph = kph
        # cruising speed in meter per second
        self.mps = kph / 3.6
        # leaving timestamp
        self.t_leave = t_leave
        # number of local data
        self.n_data = n_data
        # local dataset
        self.local_data = local_data
        # cpu frequency in hertz
        self.cpu_hz = cpu_hz
        # batch size
        self.batch = batch
        # number of local epochs
        self.epochs = epochs
        # number of cpu cycles per data
        self.cycles = cycles
        # effective capacitance
        self.eff_capa = eff_capa
        # channel signal-to-noise ratio in decibel, at distance d_ref
        self.snr_db = snr_db
        # same ratio in linear scale
        self.snr_lin = 10 ** (snr_db / 10)
        # allocated bandwidth
        self.bw_hz = bw_hz
        # client transmission power in watts
        self.ptx = ptx
        # learning rate
        self.lr = lr
        # sgd momentum
        self.mom = mom
        # sgd weight decay
        self.decay = decay
        # trajectory along the road (None: position is ignored)
        self.track = track
        # path loss exponent (0: SNR does not depend on distance)
        self.ple = ple
        # reference distance of snr_db, in meters
        self.d_ref = d_ref

    def get_cp_time(self) -> float:
        """Compute client computation time (seconds)."""
        return (self.cycles * self.n_data * self.epochs) / self.cpu_hz

    def get_cp_energy(self) -> float:
        """Compute client computation energy (joules)."""
        return self.get_cp_time() * self.eff_capa * (self.cpu_hz**3)

    def get_snr_lin(self, t: float = 0.0) -> float:
        """Signal-to-noise ratio (linear) at the absolute time t."""
        if self.track is None or self.ple == 0:
            return float(self.snr_lin)
        d = self.track.distance(t - self.t_arrive)
        loss_db = 10 * self.ple * np.log10(d / self.d_ref)
        return float(10 ** ((self.snr_db - loss_db) / 10))

    def get_throughput(self, t: float = 0.0) -> float:
        """Compute client uplink throughput (bit per second) at time t."""
        return float(self.bw_hz * np.log2(1 + self.get_snr_lin(t)))

    def get_co_time(self, m_size: int, m_prec: int, t: float = 0.0) -> float:
        """Compute communication time (s) of an upload starting at t."""
        return m_size * m_prec / self.get_throughput(t)

    def get_co_energy(self, m_size: int, m_prec: int, t: float = 0.0) -> float:
        """Compute communication energy (J) of an upload starting at t."""
        return self.ptx * self.get_co_time(m_size, m_prec, t)

    def can_finish(self, t_now: float, m_size: int, m_prec: int) -> bool:
        """Tell if the client can train and upload before leaving."""
        cp = self.get_cp_time()
        need = cp + self.get_co_time(m_size, m_prec, t_now + cp)
        return need <= self.t_leave - t_now

    def local_update(self, gmodel: nn.Module, dev: torch.device) -> nn.Module:
        """Run local training starting from the global model."""
        loader = DataLoader(
            self.local_data, batch_size=self.batch, shuffle=True
        )
        model = copy.deepcopy(gmodel).to(dev)
        crit = nn.CrossEntropyLoss()
        opt = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.mom,
            weight_decay=self.decay,
        )
        model.train()
        for _ in range(self.epochs):
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(dev), lbls.to(dev)
                loss = crit(model(imgs), lbls)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model


def get_arrivals(sim_time: float, rate: float) -> list[float]:
    """Compute all clients arrival timestamps (Poisson process)."""
    arr: list[float] = []
    t = float(np.random.exponential(1 / rate))
    while t < sim_time:
        arr.append(t)
        t += float(np.random.exponential(1 / rate))
    return arr


def get_all_clients(
    arr: list[float],
    trainset: Dataset,
    cls_idx: dict[int, list[int]],
    cfg: Config,
) -> list[Client]:
    """Create all clients."""
    clients: list[Client] = []
    kmh = 1 / 3.6  # km/h -> m/s
    for i, t_arrive in enumerate(arr):
        kph = float(np.random.uniform(cfg.min_speed_kph, cfg.max_speed_kph))
        speeds = speed_profile(
            kph * kmh,
            cfg.min_speed_kph * kmh,
            cfg.max_speed_kph * kmh,
            cfg.speed_alpha,
            cfg.speed_std_kph * kmh,
            cfg.road_length_m,
            cfg.speed_step_s,
        )
        t_leave = t_arrive + stay_time(
            speeds, cfg.speed_step_s, cfg.road_length_m
        )
        track = make_track(
            speeds, cfg.speed_step_s, cfg.road_length_m, cfg.bs_offset_m
        )
        n_wanted = int(np.random.randint(cfg.min_n_data, cfg.max_n_data + 1))
        local = sample_local_data(
            trainset, cls_idx, n_wanted, cfg.n_sub_classes
        )
        cpu_hz = float(np.random.uniform(cfg.min_cpu_hertz, cfg.max_cpu_hertz))
        snr_db = float(np.random.uniform(cfg.snr_db_min, cfg.snr_db_max))
        clients.append(
            Client(
                i,
                t_arrive,
                kph,
                t_leave,
                len(local),
                local,
                cpu_hz,
                cfg.batch,
                cfg.n_local_epochs,
                cfg.n_cpu_cycles_per_data,
                cfg.effective_capacitance,
                snr_db,
                cfg.bandwidth_hz,
                cfg.tx_power_w,
                cfg.learning_rate,
                cfg.momentum,
                cfg.weight_decay,
                track=track,
                ple=cfg.path_loss_exp,
                d_ref=cfg.ref_distance_m,
            )
        )
    return clients
