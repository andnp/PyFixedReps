import unittest
import numpy as np
from PyFixedReps import TileCoder, TileCoderConfig

class TestTileCoder(unittest.TestCase):
    def test_get_indices_1d_1tiling(self):
        config = TileCoderConfig(
            tiles=2,
            tilings=1,
            dims=1,
        )
        tc = TileCoder(config)

        indices = tc.get_indices([0])
        self.assertListEqual(list(indices), [0])

        indices = tc.get_indices([0.4])
        self.assertListEqual(list(indices), [0])

        indices = tc.get_indices([0.5])
        self.assertListEqual(list(indices), [1])

        indices = tc.get_indices([1.0])
        self.assertListEqual(list(indices), [1])

    def test_get_indices_2d_1tiling(self):
        config = TileCoderConfig(
            dims=2,
            tilings=1,
            tiles=2,
        )
        tc = TileCoder(config)

        indices = tc.get_indices([0, 0])
        self.assertListEqual(list(indices), [0])

        indices = tc.get_indices([0.99, 0])
        self.assertListEqual(list(indices), [1])

        indices = tc.get_indices([0, 0.99])
        self.assertListEqual(list(indices), [2])

        indices = tc.get_indices([0.6, 0.8])
        self.assertListEqual(list(indices), [3])

    def test_get_indices_1d_2tiling(self):
        config = TileCoderConfig(
            dims=1,
            tilings=2,
            tiles=2,
        )
        tc = TileCoder(config)

        indices = tc.get_indices([0])
        self.assertListEqual(list(indices), [0, 2])

        indices = tc.get_indices([0.99])
        self.assertListEqual(list(indices), [1, 3])

        indices = tc.get_indices([.3])
        self.assertListEqual(list(indices), [0, 3])

        indices = tc.get_indices([.51])
        self.assertListEqual(list(indices), [1, 3])

    def test_get_indices_2d_2tiling(self):
        config = TileCoderConfig(
            bound='clip',
            dims=2,
            tilings=2,
            tiles=2,
        )
        tc = TileCoder(config)

        indices = tc.get_indices([0, 0])
        self.assertListEqual(list(indices), [0, 4])

        indices = tc.get_indices([0.99, 0.99])
        self.assertListEqual(list(indices), [3, 7])

        indices = tc.get_indices([.3, .2])
        self.assertListEqual(list(indices), [0, 5])

        indices = tc.get_indices([.51, .51])
        self.assertListEqual(list(indices), [3, 7])

    def test_get_indices_2d_2tiling_random(self):

        config = TileCoderConfig(
            dims=2,
            tilings=2,
            tiles=2,
            offset='random',
        )

        rng = np.random.RandomState(42)
        tc = TileCoder(config, rng=rng)

        indices = tc.get_indices([0, 0])
        self.assertListEqual(list(indices), [2, 7])

    def test_encode(self):
        config = TileCoderConfig(
            dims=2,
            tilings=2,
            tiles=2,
            scale_output=False,
        )
        tc = TileCoder(config)

        rep = tc.encode([0, 0.2])
        expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        self.assertListEqual(list(rep), expected)

    def test_scaling(self):
        config = TileCoderConfig(
            dims=2,
            tilings=2,
            tiles=2,
            input_ranges=[(-1, 1), (2.1, 4.1)],
        )
        tc = TileCoder(config)

        rep = tc.encode([-1, 2.5])
        expected = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        self.assertListEqual(list(rep), expected)

    def test_tabular(self):
        config = TileCoderConfig(
            dims=2,
            tiles=7,
            tilings=1,
            input_ranges=[(0, 7), (0, 7)],
        )
        tc = TileCoder(config)

        out = []
        for i in range(7):
            for j in range(7):
                rep = tc.encode([j, i])
                out.append(rep)

        self.assertTrue(np.allclose(
            np.stack(out),
            np.eye(7 * 7),
        ))
