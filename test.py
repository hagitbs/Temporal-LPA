from typing import Literal

import numpy as np

from algo import entropy

Bounds = Literal["lowest", "highest"]


def check_epsilon_bounds(bounds: Bounds, matrix: np.array):
    entropy_ = entropy(matrix).sum(axis=1)
    if bounds == "lowest":
        return entropy_.min()
    elif bounds == "highest":
        ker = np.count_nonzero((matrix == 0), axis=1).max()
        return entropy_.max() + ker + 1 / (np.e * np.log(2))
