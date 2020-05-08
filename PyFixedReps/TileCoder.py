import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation

class Tiling(BaseRepresentation):
    def __init__(self, params):
        # number of dimensions to tile together
        self.dims = params['dims']
        # number of tiles for each dimension
        self.tiles_per_dim = params['tiles']
        # total number of tiles
        self.tiles = self.tiles_per_dim ** self.dims
        # length of a tile in each dimension, small offset for numerical stability
        self.tile_length = (1.0 / self.tiles_per_dim) + 1e-12
        # precomputed index offset for each dimension
        self.dim_offset = np.array([self.tiles_per_dim**d for d in range(self.dims)])

    def get_index(self, pos):
        ind = np.sum((pos + 1e-12) // self.tile_length * self.dim_offset)
        return ind % self.tiles

    def features(self):
        return int(self.tiles)

    def encode(self, s, a = None):
        index = self.get_index(s)
        vec = np.zeros(self.features())
        vec[index] = 1
        return vec

class TileCoder(BaseRepresentation):
    def __init__(self, params):
        self.num_tiling = params['tilings']

        self.num_action = params.get('actions', 1)
        self.random_offset = params.get('random_offset', False)

        self.tilings = [ Tiling(params) for _ in range(self.num_tiling) ]
        self.tiling_offsets = [ self._build_offset(ntl) for ntl in range(self.num_tiling) ]

        self.total_tiles = sum(map(lambda t: t.features(), self.tilings))

    # construct tiling offsets
    # defaults to evenly space tilings
    def _build_offset(self, n):
        dims = self.tilings[n].dims
        tile_length = self.tilings[n].tile_length
        if self.random_offset:
            return np.random.uniform(0, 1, size = dims)

        return np.ones(dims) * n * (tile_length / self.num_tiling)

    def get_indices(self, pos, action=None):
        pos = np.array(pos)
        index = np.zeros((self.num_tiling))
        for ntl in range(self.num_tiling):
            tiling = self.tilings[ntl]
            ind = tiling.get_index(pos + self.tiling_offsets[ntl])
            index[ntl] = ind + tiling.tiles * ntl

        if action != None:
            index += action * self.total_tiles

        return index.astype(int)

    def features(self):
        return int(self.total_tiles * self.num_action)

    def encode(self, s, a = None):
        indices = self.get_indices(s, a)
        vec = np.zeros(self.features())
        vec[indices] = 1
        return vec

class ScaledTileCoder(TileCoder):
    def __init__(self, dim_ranges):
        self.dim_ranges = np.array(dim_ranges)

    def encode(self, pos, action=None):
        pos = np.array(pos)
        scaled = (pos + self.dim_ranges[:, 0]) / (self.dim_ranges[:, 1] - self.dim_ranges[:, 0])

        return super().encode(scaled, action)
