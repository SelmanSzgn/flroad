import torch


def aggregate(gmodel, ups):
    state = gmodel.state_dict()
    with torch.no_grad():
        for k in state.keys():
            state[k] = sum(w * m.state_dict()[k] for w, m in ups)
    gmodel.load_state_dict(state)
    return gmodel
