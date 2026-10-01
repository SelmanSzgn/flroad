import torch
import torch.nn as nn

from flroad.server import aggregate


def linear_with(value):
    m = nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.fill_(value)
        m.bias.fill_(value)
    return m


def test_weighted_average():
    # (1 * 1 + 3 * 3) / 4 = 2.5
    g = aggregate(
        linear_with(0.0), [linear_with(1.0), linear_with(3.0)], [1, 3]
    )
    assert torch.allclose(g.weight, torch.full((1, 2), 2.5))
    assert torch.allclose(g.bias, torch.full((1,), 2.5))


def test_sizes_are_normalized():
    a = aggregate(
        linear_with(0.0), [linear_with(1.0), linear_with(3.0)], [1, 3]
    )
    b = aggregate(
        linear_with(0.0), [linear_with(1.0), linear_with(3.0)], [10, 30]
    )
    assert torch.allclose(a.weight, b.weight)


def test_empty_returns_global_model_unchanged():
    g = linear_with(0.7)
    out = aggregate(g, [], [])
    assert out is g
    assert torch.allclose(out.weight, torch.full((1, 2), 0.7))
    assert torch.allclose(out.bias, torch.full((1,), 0.7))


def test_integer_buffers_keep_their_dtype():
    g = aggregate(nn.BatchNorm1d(2), [nn.BatchNorm1d(2)] * 2, [1, 1])
    assert g.num_batches_tracked.dtype == torch.long
