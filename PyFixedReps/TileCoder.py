import numpy as np
from typing import Optional, Sequence
from numba import njit
from PyFixedReps.BaseRepresentation import BaseRepresentation

@njit(cache=True)
def tileLength(tiles_per_dim: int):
    return 1.0 / tiles_per_dim + 1e-12

@njit(cache=True)
def getTilingIndex(dims: int, tiles_per_dim: int, pos: Sequence[float]):
    ind = 0

    tile_length = tileLength(tiles_per_dim)
    total_tiles = tiles_per_dim ** dims
    for d in range(dims):
        ind += (pos[d] + 1e-12) // tile_length * tiles_per_dim**d

    return ind % total_tiles

@njit(cache=True)
def getTCIndices(dims: int, tiles: int, tilings: int, offsets: np.ndarray, pos: np.ndarray, action: Optional[int] = None):
    total_tiles = tiles**dims

    index = np.empty((tilings), dtype='int64')
    for ntl in range(tilings):
        ind = getTilingIndex(dims, tiles, pos + offsets[ntl])
        index[ntl] = ind + total_tiles * ntl

    if action != None:
        index += action * total_tiles * tilings

    return index

@njit(cache=True)
def minMaxScaling(x: np.ndarray, mi: np.ndarray, ma: np.ndarray):
    return (x - mi) / (ma - mi)

class TileCoder(BaseRepresentation):
    def __init__(self, params, rng=np.random):
        self.random = rng
        self.num_tiling:int = params['tilings']

        self.num_action:int = params.get('actions', 1)
        self.random_offset:bool = params.get('random_offset', False)
        self.input_ranges:Optional[np.ndarray] = params.get('input_ranges')
        self.scale_output:bool = params.get('scale_output', True)

        if self.input_ranges is not None:
            self.input_ranges = np.array(self.input_ranges)

        self.dims:int = params['dims']
        self.tiles:int = params['tiles']
        self.tile_length:float = 1.0 / self.tiles + 1e-12

        self.tiling_offsets:np.ndarray = np.array([ self._build_offset(ntl) for ntl in range(self.num_tiling) ])

        self.total_tiles:int = self.num_tiling * self.tiles ** self.dims

    # construct tiling offsets
    # defaults to evenly space tilings
    def _build_offset(self, n:int):
        if self.random_offset:
            return self.random.uniform(0, 1, size = self.dims)

        return np.ones(self.dims) * n * (self.tile_length / self.num_tiling)

    def get_indices(self, pos, action=None):
        pos = np.array(pos)
        if self.input_ranges is not None:
            pos = minMaxScaling(pos, self.input_ranges[:, 0], self.input_ranges[:, 1])

        return getTCIndices(self.dims, self.tiles, self.num_tiling, self.tiling_offsets, pos, action)

    def features(self):
        return int(self.total_tiles * self.num_action)

    def encode(self, s, a = None):
        indices = self.get_indices(s, a)
        vec = np.zeros(self.features())
        vec[indices] = 1

        if self.scale_output:
            vec /= float(self.num_tiling)

        return vec
