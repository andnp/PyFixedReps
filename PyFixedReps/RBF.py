import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation

class RBF(BaseRepresentation):
    def __init__(self, params):
        self.centers = np.array(params['centers'])
        self.width = params['width']

    def features(self):
        return len(self.centers)

    def encode(self, s, a = None):
        features = np.zeros(self.features())
        diff = s - self.centers
        squared = np.sum(diff * diff, axis = 1)
        features = np.exp(-1 * squared / np.square(self.width))

        return features
