import random

import torch
import torchvision
import torchvision.transforms as transforms


def create_class_indices(trainset):
    cls_idx = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(trainset):
        cls_idx[label].append(idx)
    return cls_idx


def sample_local_data(trainset, cls_idx, n_data, n_cls):
    sel = random.sample(range(10), n_cls)
    avail = []
    for c in sel:
        avail.extend(cls_idx[c])
    if len(avail) < n_data:
        samp = avail
    else:
        samp = random.sample(avail, n_data)
    return torch.utils.data.Subset(trainset, samp)


def get_transform_train():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_trainset():
    return torchvision.datasets.CIFAR10(
        root="data/cifar10",
        train=True,
        download=True,
        transform=get_transform_train(),
    )


def get_transform_test():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_testset():
    return torchvision.datasets.CIFAR10(
        root="data/cifar10",
        train=False,
        download=False,
        transform=get_transform_test(),
    )


def get_test_loader():
    return torch.utils.data.DataLoader(
        get_testset(), batch_size=64, shuffle=False, num_workers=0
    )
