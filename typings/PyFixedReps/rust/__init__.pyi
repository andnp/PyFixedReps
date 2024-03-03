import numpy as np
from typing import Sequence

def get_tc_indices(
    dims: int,
    tiles: np.ndarray,
    tilings: int,
    bound: np.ndarray,
    offsets: np.ndarray,
    bound_strats: Sequence[str],
    pos: np.ndarray,
) -> np.ndarray: ...
