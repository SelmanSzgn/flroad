import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from flroad.client import Client


def evaluate(
    loader: DataLoader, dev: torch.device, model: nn.Module
) -> tuple[float, float]:
    """Return (accuracy in %, mean loss) of model over the loader."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    correct, tot_loss, tot_n = 0, 0.0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            out = model(imgs)
            tot_loss += crit(out, lbls).item()
            correct += (out.argmax(1) == lbls).sum().item()
            tot_n += lbls.size(0)
    return 100 * correct / tot_n, tot_loss / tot_n


def evaluate_clients(
    active: list[Client],
    plain_trainset: Dataset,
    model: nn.Module,
    dev: torch.device,
    batch: int = 64,
) -> tuple[list[float], list[float]]:
    """Evaluate the model on each client's data, without augmentation."""
    accs: list[float] = []
    losses: list[float] = []
    for cl in active:
        sub = Subset(plain_trainset, cl.local_data.indices)
        loader = DataLoader(sub, batch_size=batch, shuffle=False)
        acc, loss = evaluate(loader, dev, model)
        accs.append(acc)
        losses.append(loss)
    return accs, losses
