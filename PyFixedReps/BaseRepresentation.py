import numpy as np
import numpy.typing as npt
from typing import Optional

Array = npt.NDArray[np.float_]

class BaseRepresentation:
    def encode(self, s: npt.ArrayLike, a: Optional[int] = None):
        raise NotImplementedError()
