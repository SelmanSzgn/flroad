import torch
import torch.nn as nn


def evaluate(loader, dev, model):
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


def evaluate_clients(active, plain_trainset, model, dev, batch=64):
    """Evaluate the model on each client's data, without augmentation."""
    accs, losses = [], []
    for cl in active:
        sub = torch.utils.data.Subset(plain_trainset, cl.local_data.indices)
        loader = torch.utils.data.DataLoader(
            sub, batch_size=batch, shuffle=False
        )
        acc, loss = evaluate(loader, dev, model)
        accs.append(acc)
        losses.append(loss)
    return accs, losses
