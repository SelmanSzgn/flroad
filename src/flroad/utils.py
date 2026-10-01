import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all random number generators used by the simulator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
