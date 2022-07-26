import numpy as np
from PyFixedReps.TileCoder import TileCoder

tc = TileCoder({
    'dims': 2,
    'tiles': 7,
    'tilings': 1,
    'input_ranges': [(0, 6), (0, 6)],
})

out = []
for i in range(7):
    for j in range(7):
        rep = tc.encode([j, i])
        out.append(rep)

print(np.stack(out))
