import numpy as np

class BaseRepresentation:
    def encode(self, s: np.ndarray):
        raise NotImplementedError()
