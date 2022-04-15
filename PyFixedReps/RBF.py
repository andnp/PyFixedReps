import numpy as np
import numpy.typing as npt
from typing import Optional
from PyFixedReps._jit import try2jit
from PyFixedReps.BaseRepresentation import Array, BaseRepresentation

@try2jit
def gaussian_dist(x: Array, centers: Array, width: float):
    diff = x - centers
    squared = np.sum(np.square(diff), axis=1)
    features = np.exp(-1 * squared / np.square(width))
    return features

class RBF(BaseRepresentation):
    def __init__(self, params):
        self.centers = np.array(params['centers'])
        self.width = params['width']

    def features(self):
        return len(self.centers)

    def encode(self, s: npt.ArrayLike, a: Optional[int] = None):
        s = np.array(s)
        return gaussian_dist(s, self.centers, self.width)
