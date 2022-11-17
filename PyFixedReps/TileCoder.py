import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, TypeVar
from PyFixedReps._jit import try2jit
from PyFixedReps.BaseRepresentation import Array, BaseRepresentation

Range = Tuple[float, float]
RandomState = np.random.RandomState

@dataclass
class TileCoderConfig:
    tiles: int
    tilings: int
    dims: int

    offset: str = 'cascade'
    actions: int = 1
    scale_output: bool = True
    input_ranges: Optional[Sequence[Range]] = None
    bound: str = 'wrap'

class TileCoder(BaseRepresentation):
    def __init__(self, config: TileCoderConfig, rng: Optional[RandomState] = None):
        self.rng = rng
        self._c = c = config

        self._input_ranges = None
        if c.input_ranges is not None:
            self._input_ranges = np.array(c.input_ranges)

        self._tiling_offsets: Array = np.array([ self._build_offset(ntl) for ntl in range(c.tilings) ])
        self._total_tiles: int = c.tilings * c.tiles ** c.dims

    # construct tiling offsets
    # defaults to evenly spaced tilings
    def _build_offset(self, n: int):
        if self._c.offset == 'random':
            assert self.rng is not None
            return self.rng.uniform(0, 1, size=self._c.dims)

        if self._c.offset == 'cascade':
            tile_length = 1.0 / self._c.tiles
            return np.ones(self._c.dims) * n * (tile_length / self._c.tilings)

        if self._c.offset == 'even':
            tile_length = 1.0 / self._c.tiles
            i = n - (self._c.tilings / 2)
            return np.ones(self._c.dims) * i * (tile_length / self._c.tilings)

        raise Exception('Unknown offset type')

    def get_indices(self, pos: npt.ArrayLike, action: Optional[int] = None):
        pos_: Array = np.asarray(pos, dtype=np.float_)
        if self._input_ranges is not None:
            pos_ = minMaxScaling(pos_, self._input_ranges[:, 0], self._input_ranges[:, 1])

        return getTCIndices(self._c.dims, self._c.tiles, self._c.tilings, self._c.bound, self._tiling_offsets, pos_, action)

    def features(self):
        return int(self._total_tiles * self._c.actions)

    def encode(self, s: npt.ArrayLike, a: Optional[int] = None):
        indices = self.get_indices(s, a)
        vec = np.zeros(self.features())

        v = 1.
        if self._c.scale_output:
            v = 1. / self._c.tilings

        vec[indices] = v
        return vec

@try2jit
def getAxisCell(bound: str, x: float, tiles: int):
    # for a 2-d space, this would get the "row" then "col" for a given coordinate
    # for example: pos = (0.1, 0.3) with 5 tiles per dim would give row=1 and col=2
    i = int(np.floor(x * tiles))

    if bound == 'wrap':
        return i % tiles
    elif bound == 'clip':
        return clip(i, 0, tiles - 1)

    raise Exception('Unknown bound type')

@try2jit
def getTilingIndex(bound: str, dims: int, tiles_per_dim: int, pos: Array):
    ind = 0

    total_tiles = tiles_per_dim ** dims
    for d in range(dims):
        # which cell am I in on this axis?
        axis = getAxisCell(bound, pos[d], tiles_per_dim)
        already_seen = tiles_per_dim ** d
        ind += axis * already_seen

    # ensure we don't overflow into another tiling
    return clip(ind, 0, total_tiles - 1)

@try2jit
def getTCIndices(dims: int, tiles: int, tilings: int, bound: str, offsets: Array, pos: Array, action: Optional[int] = None):
    total_tiles = tiles**dims

    index = np.empty((tilings), dtype='int64')
    for ntl in range(tilings):
        ind = getTilingIndex(bound, dims, tiles, pos + offsets[ntl])
        index[ntl] = ind + total_tiles * ntl

    if action is not None:
        index += action * total_tiles * tilings

    return index

@try2jit
def minMaxScaling(x: Array, mi: Array, ma: Array):
    return (x - mi) / (ma - mi)

T = TypeVar('T', bound=float)
@try2jit
def clip(x: T, mi: T, ma: T) -> T:
    return max(min(x, ma), mi)
