import numpy as np
from PyFixedReps import TileCoder, TileCoderConfig

def test_get_indices_1d_1tiling():
    config = TileCoderConfig(
        tiles=2,
        tilings=1,
        dims=1,
    )
    tc = TileCoder(config)

    indices = tc.get_indices([0])
    assert list(indices) == [0]

    indices = tc.get_indices([0.4])
    assert list(indices) == [0]

    indices = tc.get_indices([0.5])
    assert list(indices) == [1]

    indices = tc.get_indices([1.0])
    assert list(indices) == [1]

def test_get_indices_2d_1tiling():
    config = TileCoderConfig(
        dims=2,
        tilings=1,
        tiles=2,
    )
    tc = TileCoder(config)

    indices = tc.get_indices([0, 0])
    assert list(indices) == [0]

    indices = tc.get_indices([0.99, 0])
    assert list(indices) == [1]

    indices = tc.get_indices([0, 0.99])
    assert list(indices) == [2]

    indices = tc.get_indices([0.6, 0.8])
    assert list(indices) == [3]

def test_get_indices_1d_2tiling():
    config = TileCoderConfig(
        dims=1,
        tilings=2,
        tiles=2,
    )
    tc = TileCoder(config)

    indices = tc.get_indices([0])
    assert list(indices) == [0, 2]

    indices = tc.get_indices([0.99])
    assert list(indices) == [1, 3]

    indices = tc.get_indices([.3])
    assert list(indices) == [0, 3]

    indices = tc.get_indices([.51])
    assert list(indices) == [1, 3]

def test_get_indices_2d_2tiling():
    config = TileCoderConfig(
        bound='clip',
        dims=2,
        tilings=2,
        tiles=2,
    )
    tc = TileCoder(config)

    indices = tc.get_indices([0, 0])
    assert list(indices) == [0, 4]

    indices = tc.get_indices([0.99, 0.99])
    assert list(indices) == [3, 7]

    indices = tc.get_indices([.3, .2])
    assert list(indices) == [0, 5]

    indices = tc.get_indices([.51, .51])
    assert list(indices) == [3, 7]

def test_get_indices_2d_2tiling_random():

    config = TileCoderConfig(
        dims=2,
        tilings=2,
        tiles=2,
        offset='random',
    )

    rng = np.random.RandomState(42)
    tc = TileCoder(config, rng=rng)

    indices = tc.get_indices([0, 0])
    assert list(indices) == [2, 7]

def test_encode():
    config = TileCoderConfig(
        dims=2,
        tilings=2,
        tiles=2,
        scale_output=False,
    )
    tc = TileCoder(config)

    rep = tc.encode([0, 0.2])
    expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    assert list(rep) == expected

def test_scaling():
    config = TileCoderConfig(
        dims=2,
        tilings=2,
        tiles=2,
        input_ranges=[(-1, 1), (2.1, 4.1)],
    )
    tc = TileCoder(config)

    rep = tc.encode([-1, 2.5])
    expected = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    assert list(rep) == expected

def test_tabular():
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

    assert np.allclose(
        np.stack(out),
        np.eye(7 * 7),
    )

def test_benchmark_encode(benchmark):
    config = TileCoderConfig(
        dims=2,
        tilings=2,
        tiles=2,
        scale_output=False,
    )
    tc = TileCoder(config)
    point = [0, 0.2]

    def test(tc, point):
        tc.encode(point)

    benchmark(test, tc, point)
