import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from model import Model
from data import sample_local_data


class Client:
    def __init__(
            self, cid, t_arrive, kph, t_leave, n_data, local_data,
            cpu_hz, batch, epochs, cycles, eff_capa, snr_db, bw_hz,
            ptx, lr, mom, decay
    ):
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

    def get_cp_time(self):
        """Compute client computation time (seconds)."""
        return (self.cycles * self.n_data * self.epochs) / self.cpu_hz

    def get_cp_energy(self):
        """Compute client communication energy (joules)."""
        return self.get_cp_time() * self.eff_capa * (self.cpu_hz ** 3)

    def get_throughput(self):
        """Compute client uplink throughput (bit per second)."""
        return self.bw_hz * np.log2(1 + self.snr_lin)

    def get_co_time(self, m_size, m_prec):
        """Compute client communication time (seconds)."""
        bits = m_size * m_prec
        return bits / self.get_throughput()

    def get_co_energy(self, m_size, m_prec):
        """Compute client communication energy (joules)."""
        return self.ptx * self.get_co_time(m_size, m_prec)

    def local_update(self, gmodel, dev):
        """Run local training."""
        loader = torch.utils.data.DataLoader(
            self.local_data, batch_size=self.batch, shuffle=True
        )
        model = Model()
        model.load_state_dict(copy.deepcopy(gmodel.state_dict()))
        model = model.to(dev)
        crit = nn.CrossEntropyLoss()
        opt = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.mom,
            weight_decay=self.decay,
        )
        for _ in range(self.epochs):
            model.train()
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(dev), lbls.to(dev)
                out = model(imgs)
                loss = crit(out, lbls)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model


def get_arrivals(sim_time, rate):
    """Compute all clients arrival timestamps (Poisson process)."""
    arr = []
    t = 0
    stop = False
    while not stop:
        t += np.random.exponential(1 / rate)
        if t < sim_time:
            arr.append(t)
        else:
            stop = True
    return arr


def get_all_clients(
        arr, trainset, cls_idx, n_cls, min_kph, max_kph, road_m,
        min_n, max_n, min_cpu, max_cpu, batch, epochs, cycles,
        eff_capa, snr_min, snr_max, bw_hz, ptx, lr, mom, decay
):
    """Create all clients."""
    clients = []
    for i in range(len(arr)):
        kph = np.random.uniform(min_kph, max_kph)
        mps = kph / 3.6
        t_leave = arr[i] + road_m / mps
        n_data = np.random.randint(min_n, max_n + 1)
        local = sample_local_data(trainset, cls_idx, n_data, n_cls)
        cpu_hz = np.random.uniform(min_cpu, max_cpu)
        snr_db = np.random.uniform(snr_min, snr_max)
        clients.append(Client(
            i, arr[i], kph, t_leave, n_data, local, cpu_hz, batch,
            epochs, cycles, eff_capa, snr_db, bw_hz, ptx, lr, mom,
            decay,
        ))
    return clients
