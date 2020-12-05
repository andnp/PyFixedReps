import numpy as np
from numba import njit
from PyFixedReps.BaseRepresentation import BaseRepresentation

@njit(cache=True)
def gaussian_dist(x: np.ndarray, centers: np.ndarray, width: float):
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

    def encode(self, s, a = None):
        s = np.array(s)
        return gaussian_dist(s, self.centers, self.width)
