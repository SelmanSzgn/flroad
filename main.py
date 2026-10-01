import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import yaml

from client import Client, get_all_clients, get_arrivals
from data import get_trainset, get_test_loader, create_class_indices
from model import Model
from server import aggregate
from eval import (
    get_test_accuracy, get_test_loss, get_client_accuracy, get_client_loss
)


def ts():
    """Return the current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":

    with open("cfg.yaml", "r") as f:
        c = yaml.safe_load(f)

    np.random.seed(c["numpy_seed"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gmodel = Model()
    m_size = sum(p.numel() for p in gmodel.parameters() if p.requires_grad)

    trainset = get_trainset()
    test_loader = get_test_loader()
    cls_idx = create_class_indices(trainset)

    arr = get_arrivals(c["simulation_time_s"], c["poisson_rate"])
    clients = get_all_clients(
        arr, trainset, cls_idx, c["n_sub_classes"],
        c["min_speed_kph"], c["max_speed_kph"], c["road_length_m"],
        c["min_n_data"], c["max_n_data"],
        float(c["min_cpu_hertz"]), float(c["max_cpu_hertz"]),
        c["batch"], c["n_local_epochs"],
        float(c["n_cpu_cycles_per_data"]),
        float(c["effective_capacitance"]),
        float(c["snr_db_min"]), float(c["snr_db_max"]),
        float(c["bandwidth_hz"]), c["tx_power_w"],
        float(c["learning_rate"]), c["momentum"],
        float(c["weight_decay"]),
    )

    dur = c["round_duration_s"]
    n_rounds = int(c["simulation_time_s"] / dur)

    for r in range(n_rounds):
        n_drop = 0
        ups = []
        active = [
            cl for cl in clients
            if cl.t_arrive <= r * dur and cl.t_leave > r * dur
        ]
        n_act = len(active)
        if n_act == 0:
            print(f"[{ts()}] round {r + 1}/{n_rounds}")
            print("  No active client, continue.")
        else:
            tot = sum([cl.n_data for cl in active])
            print(f"[{ts()}] round {r + 1}/{n_rounds}")
            print(f"  Number of active clients: {n_act}")
            for cl in active:
                cp_time = cl.get_cp_time()
                cp_energy = cl.get_cp_energy()
                co_time = cl.get_co_time(m_size, c["model_precision"])
                co_energy = cl.get_co_energy(m_size, c["model_precision"])
                if cp_time + co_time <= cl.t_leave:
                    lmodel = cl.local_update(gmodel, dev)
                    ups.append((cl.n_data / tot, lmodel))
                else:
                    ups.append((cl.n_data / tot, gmodel))
                    n_drop += 1
            gmodel = aggregate(gmodel, ups)

            test_acc = get_test_accuracy(test_loader, dev, gmodel)
            test_loss = get_test_loss(test_loader, dev, gmodel)
            cl_acc = get_client_accuracy(active, gmodel, dev)
            cl_loss = get_client_loss(active, gmodel, dev)

            print(f"  Test accuracy: {test_acc:3g} %")
