import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float_]

class BaseRepresentation:
    def encode(self, s: npt.ArrayLike):
        raise NotImplementedError()
