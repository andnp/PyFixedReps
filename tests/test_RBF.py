import numpy as np
from PyFixedReps.RBF import RBF

def test_encode():
    rbf = RBF({
        'centers': [[0, 0], [1, 0], [0, 1], [1, 1]],
        'width': 2,
    })

    rep = rbf.encode([0, 0.2])
    expected = [0.9900498337491681, 0.7710515858035663, 0.8521437889662113, 0.6636502501363194]

    assert np.all(np.isclose(rep, expected))
