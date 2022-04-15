import numpy as np
import numpy.typing as npt
from typing import Optional
from PyFixedReps._jit import try2jit
from PyFixedReps.BaseRepresentation import Array, BaseRepresentation

@try2jit
def tileLength(tiles_per_dim: int):
    return 1.0 / tiles_per_dim + 1e-16

@try2jit
def getTilingIndex(dims: int, tiles_per_dim: int, pos: Array):
    ind = 0

    tile_length = tileLength(tiles_per_dim)
    total_tiles = tiles_per_dim ** dims
    for d in range(dims):
        ind += (pos[d] + 1e-16) // tile_length * tiles_per_dim**d

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

class TileCoder(BaseRepresentation):
    def __init__(self, params, rng=np.random):
        self.random = rng
        self.num_tiling: int = params['tilings']

        self.num_action: int = params.get('actions', 1)
        self.random_offset: bool = params.get('random_offset', False)
        self.input_ranges: Optional[Array] = params.get('input_ranges')
        self.scale_output: bool = params.get('scale_output', True)

        if self.input_ranges is not None:
            self.input_ranges = np.array(self.input_ranges)

        self.dims: int = params['dims']
        self.tiles: int = params['tiles']
        self.tile_length: float = tileLength(self.tiles)

        self.tiling_offsets: Array = np.array([ self._build_offset(ntl) for ntl in range(self.num_tiling) ])

        self.total_tiles: int = self.num_tiling * self.tiles ** self.dims

    # construct tiling offsets
    # defaults to evenly space tilings
    def _build_offset(self, n: int):
        if self.random_offset:
            return self.random.uniform(0, 1, size=self.dims)

        return np.ones(self.dims) * n * (self.tile_length / self.num_tiling)

    def get_indices(self, pos: npt.ArrayLike, action: Optional[int] = None):
        pos_: Array = np.array(pos, dtype=np.float_)
        if self.input_ranges is not None:
            pos = minMaxScaling(pos_, self.input_ranges[:, 0], self.input_ranges[:, 1])

        return getTCIndices(self.dims, self.tiles, self.num_tiling, self.tiling_offsets, pos_, action)

    def features(self):
        return int(self.total_tiles * self.num_action)

    def encode(self, s: npt.ArrayLike, a: Optional[int] = None):
        indices = self.get_indices(s, a)
        vec = np.zeros(self.features())
        vec[indices] = 1

        if self.scale_output:
            vec /= float(self.num_tiling)

        return vec
