import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from flroad.config import Config
from flroad.data import sample_local_data
from flroad.mobility import speed_profile, stay_time


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
    ) -> None:
        # client id
        self.cid = cid
        # arrival timestamp
        self.t_arrive = t_arrive
        # speed in kilometer per hour
        self.kph = kph
        # speed in meter per second
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
        # channel signal-to-noise ratio in decibel
        self.snr_db = snr_db
        # channel signal-to-noise ratio in linear scale
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

    def get_cp_time(self) -> float:
        """Compute client computation time (seconds)."""
        return (self.cycles * self.n_data * self.epochs) / self.cpu_hz

    def get_cp_energy(self) -> float:
        """Compute client computation energy (joules)."""
        return self.get_cp_time() * self.eff_capa * (self.cpu_hz**3)

    def get_throughput(self) -> float:
        """Compute client uplink throughput (bit per second)."""
        return float(self.bw_hz * np.log2(1 + self.snr_lin))

    def get_co_time(self, m_size: int, m_prec: int) -> float:
        """Compute client communication time (seconds)."""
        return m_size * m_prec / self.get_throughput()

    def get_co_energy(self, m_size: int, m_prec: int) -> float:
        """Compute client communication energy (joules)."""
        return self.ptx * self.get_co_time(m_size, m_prec)

    def can_finish(self, t_now: float, m_size: int, m_prec: int) -> bool:
        """Tell if the client can train and upload before leaving."""
        need = self.get_cp_time() + self.get_co_time(m_size, m_prec)
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
            )
        )
    return clients
