import random
from typing import Any

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

NORM = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def create_class_indices(
    trainset: CIFAR10, n_cls: int = 10
) -> dict[int, list[int]]:
    """Map each class label to the list of its sample indices."""
    cls_idx: dict[int, list[int]] = {i: [] for i in range(n_cls)}
    for idx, label in enumerate(trainset.targets):
        cls_idx[int(label)].append(idx)
    return cls_idx


def sample_local_data(
    trainset: Dataset,
    cls_idx: dict[int, list[int]],
    n_data: int,
    n_cls: int,
) -> Subset:
    """Draw up to n_data samples from n_cls randomly chosen classes."""
    sel = random.sample(sorted(cls_idx), n_cls)
    avail: list[int] = []
    for c in sel:
        avail.extend(cls_idx[c])
    if len(avail) < n_data:
        samp = avail
    else:
        samp = random.sample(avail, n_data)
    return Subset(trainset, samp)


def get_transform(augment: bool) -> T.Compose:
    """Return the image transform, with or without augmentation."""
    ops: list[Any] = []
    if augment:
        ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    return T.Compose(ops + [T.ToTensor(), NORM])


def get_trainset(root: str, augment: bool = True) -> CIFAR10:
    return CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=get_transform(augment),
    )


def get_testset(root: str) -> CIFAR10:
    return CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=get_transform(False),
    )


def get_test_loader(root: str) -> DataLoader:
    return DataLoader(
        get_testset(root), batch_size=64, shuffle=False, num_workers=0
    )
