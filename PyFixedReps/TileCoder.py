import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from PyFixedReps._jit import try2jit
from PyFixedReps.BaseRepresentation import Array, BaseRepresentation

Range = Tuple[float, float]

@dataclass
class TileCoderConfig:
    tiles: int
    tilings: int
    dims: int

    offset: str = 'even'
    actions: int = 1
    scale_output: bool = True
    input_ranges: Optional[Sequence[Range]] = None


class TileCoder(BaseRepresentation):
    def __init__(self, config: TileCoderConfig, rng=np.random):
        self.random = rng
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
            return self.random.uniform(0, 1, size=self._c.dims)

        tile_length = 1.0 / self._c.tiles
        return np.ones(self._c.dims) * n * (tile_length / self._c.tilings)

    def get_indices(self, pos: npt.ArrayLike, action: Optional[int] = None):
        pos_: Array = np.asarray(pos, dtype=np.float_)
        if self._input_ranges is not None:
            pos_ = minMaxScaling(pos_, self._input_ranges[:, 0], self._input_ranges[:, 1])

        return getTCIndices(self._c.dims, self._c.tiles, self._c.tilings, self._tiling_offsets, pos_, action)

    def features(self):
        return int(self._total_tiles * self._c.actions)

    def encode(self, s: npt.ArrayLike, a: Optional[int] = None):
        indices = self.get_indices(s, a)
        vec = np.zeros(self.features())
        vec[indices] = 1

        if self._c.scale_output:
            vec /= float(self._c.tilings)

        return vec

@try2jit
def getAxisCell(x: float, tiles: int):
    # for a 2-d space, this would get the "row" then "col" for a given coordinate
    # for example: pos = (0.1, 0.3) with 5 tiles per dim would give row=1 and col=2

    return int(np.floor(x * tiles))

@try2jit
def getTilingIndex(dims: int, tiles_per_dim: int, pos: Array):
    ind = 0

    total_tiles = tiles_per_dim ** dims
    for d in range(dims):
        # which cell am I in on this axis?
        axis = getAxisCell(pos[d], tiles_per_dim)
        already_seen = tiles_per_dim ** d
        ind += axis * already_seen

    # ensure we don't accidentally overflow into another tiling
    return ind % total_tiles

@try2jit
def getTCIndices(dims: int, tiles: int, tilings: int, offsets: Array, pos: Array, action: Optional[int] = None):
    total_tiles = tiles**dims

    index = np.empty((tilings), dtype='int64')
    for ntl in range(tilings):
        ind = getTilingIndex(dims, tiles, pos + offsets[ntl])
        index[ntl] = ind + total_tiles * ntl

    if action is not None:
        index += action * total_tiles * tilings

    return index

@try2jit
def minMaxScaling(x: Array, mi: Array, ma: Array):
    return (x - mi) / (ma - mi)
