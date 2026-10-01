import torch


def aggregate(gmodel, models, sizes):
    """FedAvg: average client models weighted by their data sizes.

    If no client model is available, the global model is returned as is.
    """
    if not models:
        return gmodel
    tot = float(sum(sizes))
    states = [m.state_dict() for m in models]
    new = {}
    with torch.no_grad():
        for k, ref in gmodel.state_dict().items():
            if ref.is_floating_point():
                new[k] = sum(
                    (n / tot) * s[k] for n, s in zip(sizes, states)
                )
            else:
                new[k] = states[0][k].clone()
    gmodel.load_state_dict(new)
    return gmodel
