from datetime import datetime

import mlflow
import torch
import torch.nn as nn

from flroad.client import get_all_clients, get_arrivals
from flroad.config import Config
from flroad.data import create_class_indices, get_test_loader, get_trainset
from flroad.eval import evaluate, evaluate_clients
from flroad.model import Model
from flroad.runs import MetricsWriter, make_run_dir, save_config, save_model
from flroad.server import aggregate
from flroad.tracking import log_config, log_round, setup_tracking
from flroad.utils import set_seed


def ts() -> str:
    """Return the current timestamp as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(cfg: Config, experiment: str = "flroad") -> None:
    """Run one simulation and track it in mlflow."""
    setup_tracking(experiment)
    with mlflow.start_run(run_name=f"seed{cfg.seed}"):
        log_config(cfg)
        _simulate(cfg)


def _simulate(cfg: Config) -> None:
    """Run one simulation described by the validated Config cfg."""
    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir)
    print(f"[{ts()}] run folder: {run_dir}")
    writer = MetricsWriter(run_dir)

    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gmodel: nn.Module = Model().to(dev)
    m_size = sum(p.numel() for p in gmodel.parameters() if p.requires_grad)
    prec = cfg.model_precision

    trainset = get_trainset(cfg.data_path)
    plain = get_trainset(cfg.data_path, augment=False)
    test_loader = get_test_loader(cfg.data_path)
    cls_idx = create_class_indices(trainset)

    arr = get_arrivals(cfg.simulation_time_s, cfg.poisson_rate)
    clients = get_all_clients(
        arr,
        trainset,
        cls_idx,
        cfg.n_sub_classes,
        cfg.min_speed_kph,
        cfg.max_speed_kph,
        cfg.road_length_m,
        cfg.min_n_data,
        cfg.max_n_data,
        cfg.min_cpu_hertz,
        cfg.max_cpu_hertz,
        cfg.batch,
        cfg.n_local_epochs,
        cfg.n_cpu_cycles_per_data,
        cfg.effective_capacitance,
        cfg.snr_db_min,
        cfg.snr_db_max,
        cfg.bandwidth_hz,
        cfg.tx_power_w,
        cfg.learning_rate,
        cfg.momentum,
        cfg.weight_decay,
    )

    dur = cfg.round_duration_s
    n_rounds = int(cfg.simulation_time_s / dur)

    acc, loss = evaluate(test_loader, dev, gmodel)
    print(f"[{ts()}] initial model | acc {acc:.2f} % | loss {loss:.4f}")
    writer.write(
        {
            "round": 0,
            "time_s": 0,
            "active": 0,
            "dropped": 0,
            "acc": acc,
            "loss": loss,
            "energy_j": 0.0,
        }
    )
    log_round({"acc": acc, "loss": loss, "energy_j": 0.0}, step=0)

    for r in range(n_rounds):
        t = r * dur
        active = [cl for cl in clients if cl.t_arrive <= t < cl.t_leave]
        models: list[nn.Module] = []
        sizes: list[int] = []
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
        row: dict[str, float | int] = {
            "round": r + 1,
            "time_s": t,
            "active": len(active),
            "dropped": n_drop,
            "acc": acc,
            "loss": loss,
            "energy_j": energy,
        }
        if active:
            c_acc, _ = evaluate_clients(active, plain, gmodel, dev)
            row["client_acc_min"] = min(c_acc)
            row["client_acc_mean"] = sum(c_acc) / len(c_acc)
            print(
                f"  client acc: min {min(c_acc):.2f} %, "
                f"mean {sum(c_acc) / len(c_acc):.2f} %"
            )
        writer.write(row)
        log_round(row, step=r + 1)

    save_model(gmodel, run_dir)
    print(f"[{ts()}] model saved: {run_dir / 'model.pt'}")
