import torch
import torch.nn as nn


def get_test_accuracy(loader, dev, model):
    correct = 0
    total = 0
    ytrue = []
    ypred = []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            out = model(imgs)
            _, pred = torch.max(out, 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()
            ytrue.extend(lbls.cpu().numpy())
            ypred.extend(pred.cpu().numpy())
    return 100 * (correct / total)


def get_test_loss(loader, dev, model):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss = 0
    tot_n = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            out = model(imgs)
            loss = crit(out, lbls)
            n = lbls.size(0)
            tot_loss += loss.item() * n
            tot_n += n
    return tot_loss / tot_n


def get_client_accuracy(active, model, dev, batch=32):
    accs = []
    for cl in active:
        loader = torch.utils.data.DataLoader(
            cl.local_data, batch_size=batch, shuffle=True
        )
        accs.append(get_test_accuracy(loader, dev, model))
    return accs


def get_client_loss(active, model, dev, batch=32):
    losses = []
    for cl in active:
        loader = torch.utils.data.DataLoader(
            cl.local_data, batch_size=batch, shuffle=True
        )
        losses.append(get_test_loss(loader, dev, model))
    return losses
