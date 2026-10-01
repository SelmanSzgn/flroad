import random

import torch
from torch.utils.data import TensorDataset

from client import get_arrivals
from data import create_class_indices, sample_local_data
from utils import set_seed


class FakeSet(TensorDataset):
    """200 fake images, 20 per class, with a CIFAR-like .targets."""

    def __init__(self, n=200, n_cls=10):
        self.targets = [i % n_cls for i in range(n)]
        super().__init__(torch.zeros(n, 3, 32, 32),
                         torch.tensor(self.targets))


def test_class_indices_partition_the_dataset():
    idx = create_class_indices(FakeSet())
    assert sorted(i for v in idx.values() for i in v) == list(range(200))
    assert all(len(v) == 20 for v in idx.values())


def test_local_data_is_capped_by_available_samples():
    ds = FakeSet()
    idx = create_class_indices(ds)
    random.seed(0)
    assert len(sample_local_data(ds, idx, 10, 2)) == 10
    # 2 classes only hold 40 samples: asking for 100 returns 40
    assert len(sample_local_data(ds, idx, 100, 2)) == 40


def test_local_data_uses_at_most_n_classes():
    ds = FakeSet()
    idx = create_class_indices(ds)
    random.seed(1)
    sub = sample_local_data(ds, idx, 30, 3)
    assert len({ds.targets[i] for i in sub.indices}) <= 3


def test_sampling_is_reproducible():
    ds = FakeSet()
    idx = create_class_indices(ds)
    set_seed(5)
    a = sample_local_data(ds, idx, 30, 3).indices
    set_seed(5)
    b = sample_local_data(ds, idx, 30, 3).indices
    assert a == b


def test_arrivals_are_sorted_bounded_and_reproducible():
    set_seed(0)
    a = get_arrivals(1000, 0.1)
    set_seed(0)
    b = get_arrivals(1000, 0.1)
    assert a == b
    assert a == sorted(a)
    assert all(0 < t < 1000 for t in a)
