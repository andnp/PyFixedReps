import unittest
import numpy as np
from PyFixedReps import TileCoder

class TestTileCoder(unittest.TestCase):
    def test_get_indices_1d_1tiling(self):
        tc = TileCoder({
            'dims': 1,
            'tilings': 1,
            'tiles': 2,
            'actions': 2,
        })

        indices = tc.get_indices([0], 0)
        self.assertListEqual(list(indices), [0])

        indices = tc.get_indices([0.4], 0)
        self.assertListEqual(list(indices), [0])

        indices = tc.get_indices([0.5], 0)
        self.assertListEqual(list(indices), [1])

        indices = tc.get_indices([1.0], 0)
        self.assertListEqual(list(indices), [1])

        indices = tc.get_indices([0], 1)
        self.assertListEqual(list(indices), [2])

        indices = tc.get_indices([0.4], 1)
        self.assertListEqual(list(indices), [2])

        indices = tc.get_indices([0.5], 1)
        self.assertListEqual(list(indices), [3])

        indices = tc.get_indices([1.0], 1)
        self.assertListEqual(list(indices), [3])

        # test out of bounds
        indices = tc.get_indices([1.1], 0)
        self.assertListEqual(list(indices), [0])

        indices = tc.get_indices([1.1], 1)
        self.assertListEqual(list(indices), [2])

    def test_get_indices_2d_1tiling(self):
        tc = TileCoder({
            'dims': 2,
            'tilings': 1,
            'tiles': 2,
            'actions': 2,
        })

        indices = tc.get_indices([0, 0], 0)
        self.assertListEqual(list(indices), [0])

        indices = tc.get_indices([1, 0], 0)
        self.assertListEqual(list(indices), [1])

        indices = tc.get_indices([0, 1], 0)
        self.assertListEqual(list(indices), [2])

        indices = tc.get_indices([0.6, 0.8], 0)
        self.assertListEqual(list(indices), [3])

        indices = tc.get_indices([0, 0], 1)
        self.assertListEqual(list(indices), [4])

        indices = tc.get_indices([1, 0], 1)
        self.assertListEqual(list(indices), [5])

    def test_get_indices_1d_2tiling(self):
        tc = TileCoder({
            'dims': 1,
            'tilings': 2,
            'tiles': 2,
            'actions': 2,
        })

        indices = tc.get_indices([0], 0)
        self.assertListEqual(list(indices), [0, 2])

        indices = tc.get_indices([1], 0)
        self.assertListEqual(list(indices), [1, 2])

        indices = tc.get_indices([.3], 0)
        self.assertListEqual(list(indices), [0, 3])

        indices = tc.get_indices([.51], 0)
        self.assertListEqual(list(indices), [1, 3])

    def test_get_indices_2d_2tiling(self):
        tc = TileCoder({
            'dims': 2,
            'tilings': 2,
            'tiles': 2,
            'actions': 2,
        })

        indices = tc.get_indices([0, 0], 0)
        self.assertListEqual(list(indices), [0, 4])

        indices = tc.get_indices([1, 1], 0)
        self.assertListEqual(list(indices), [3, 6])

        indices = tc.get_indices([.3, .3], 0)
        self.assertListEqual(list(indices), [0, 7])

        indices = tc.get_indices([.51, .51], 0)
        self.assertListEqual(list(indices), [3, 7])

    def test_get_indices_2d_2tiling_random(self):
        np.random.seed(42)
        tc = TileCoder({
            'dims': 2,
            'tilings': 2,
            'tiles': 2,
            'actions': 1,
            'random_offset': True,
        })

        indices = tc.get_indices([0, 0], 0)
        self.assertListEqual(list(indices), [2, 7])

    def test_encode(self):
        tc = TileCoder({
            'dims': 2,
            'tilings': 2,
            'tiles': 2,
            'actions': 2,
            'scale_output': False,
        })

        rep = tc.encode([0, 0.2], 1)
        expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        self.assertListEqual(list(rep), expected)

    def test_scaling(self):
        tc = TileCoder({
            'dims': 2,
            'tilings': 2,
            'tiles': 2,
            'actions': 2,
            'input_ranges': [(-1, 1), (2.1, 4.1)]
        })

        rep = tc.encode([-1, 2.5], 1)
        expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        self.assertListEqual(list(rep), expected)
