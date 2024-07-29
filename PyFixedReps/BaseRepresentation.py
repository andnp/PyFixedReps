import numpy.typing as npt

class BaseRepresentation:
    def encode(self, s: npt.ArrayLike):
        raise NotImplementedError()
