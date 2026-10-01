from datetime import datetime

import torch
import yaml

from client import get_all_clients, get_arrivals
from data import get_trainset, get_test_loader, create_class_indices
from eval import evaluate, evaluate_clients
from model import Model
from server import aggregate
from utils import set_seed


def ts():
    """Return the current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    with open("cfg.yaml", "r") as f:
        c = yaml.safe_load(f)

    set_seed(c["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gmodel = Model().to(dev)
    m_size = sum(p.numel() for p in gmodel.parameters() if p.requires_grad)
    prec = c["model_precision"]

    trainset = get_trainset(c["data_path"])
    plain = get_trainset(c["data_path"], augment=False)
    test_loader = get_test_loader(c["data_path"])
    cls_idx = create_class_indices(trainset)

    arr = get_arrivals(c["simulation_time_s"], c["poisson_rate"])
    clients = get_all_clients(
        arr, trainset, cls_idx, c["n_sub_classes"],
        c["min_speed_kph"], c["max_speed_kph"], c["road_length_m"],
        c["min_n_data"], c["max_n_data"],
        c["min_cpu_hertz"], c["max_cpu_hertz"],
        c["batch"], c["n_local_epochs"], c["n_cpu_cycles_per_data"],
        c["effective_capacitance"], c["snr_db_min"], c["snr_db_max"],
        c["bandwidth_hz"], c["tx_power_w"], c["learning_rate"],
        c["momentum"], c["weight_decay"],
    )

    dur = c["round_duration_s"]
    n_rounds = int(c["simulation_time_s"] / dur)

    acc, loss = evaluate(test_loader, dev, gmodel)
    print(f"[{ts()}] initial model | acc {acc:.2f} % | loss {loss:.4f}")

    for r in range(n_rounds):
        t = r * dur
        active = [cl for cl in clients if cl.t_arrive <= t < cl.t_leave]
        models, sizes = [], []
        n_drop, energy = 0, 0.0
        for cl in active:
            if cl.can_finish(t, m_size, prec):
                models.append(cl.local_update(gmodel, dev))
                sizes.append(cl.n_data)
                energy += cl.get_cp_energy() + cl.get_co_energy(m_size, prec)
            else:
                n_drop += 1

        gmodel = aggregate(gmodel, models, sizes)

        acc, loss = evaluate(test_loader, dev, gmodel)
        print(
            f"[{ts()}] round {r + 1}/{n_rounds} | "
            f"active {len(active)} | dropped {n_drop} | "
            f"acc {acc:.2f} % | loss {loss:.4f} | "
            f"energy {energy:.3f} J"
        )
        if active:
            c_acc, _ = evaluate_clients(active, plain, gmodel, dev)
            print(
                f"  client acc: min {min(c_acc):.2f} %, "
                f"mean {sum(c_acc) / len(c_acc):.2f} %"
            )


if __name__ == "__main__":
    main()
