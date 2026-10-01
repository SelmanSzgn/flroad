import random

import torch
import torchvision
import torchvision.transforms as T

NORM = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def create_class_indices(trainset, n_cls=10):
    """Map each class label to the list of its sample indices."""
    cls_idx = {i: [] for i in range(n_cls)}
    for idx, label in enumerate(trainset.targets):
        cls_idx[int(label)].append(idx)
    return cls_idx


def sample_local_data(trainset, cls_idx, n_data, n_cls):
    """Draw up to n_data samples from n_cls randomly chosen classes."""
    sel = random.sample(sorted(cls_idx), n_cls)
    avail = []
    for c in sel:
        avail.extend(cls_idx[c])
    if len(avail) < n_data:
        samp = avail
    else:
        samp = random.sample(avail, n_data)
    return torch.utils.data.Subset(trainset, samp)


def get_transform(augment):
    """Return the image transform, with or without augmentation."""
    ops = []
    if augment:
        ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    return T.Compose(ops + [T.ToTensor(), NORM])


def get_trainset(root, augment=True):
    return torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=get_transform(augment),
    )


def get_testset(root):
    return torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=get_transform(False),
    )


def get_test_loader(root):
    return torch.utils.data.DataLoader(
        get_testset(root), batch_size=64, shuffle=False, num_workers=0
    )
