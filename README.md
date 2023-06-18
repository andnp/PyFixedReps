# PyFixedReps
Short for: Python Fixed Representations.
This is a collection of unit tested implementations of common fixed representations commonly used with linear (in features) RL systems.

## Installing
Can be installed using `pip` by including this in your `requirements.txt`:
```
pip install PyFixedReps-andnp==4.0.2
```
I highly recommend specifying the version number when installing in order to ensure reproducibility of experiments.
This library is fairly stable, so does not change often and there is little risk of missing an important change.

## Tile-coder
```python
from PyFixedReps import TileCoder

tc = TileCoder({
    # [required]
    'tiles': 2, # how many tiles in each tiling
    'tilings': 4,
    'dims': 2, # shape of the state-vector

    # [optional]
    'random_offset': True, # instead of using the default fixed-width offset, randomly generate offsets between tilings
    'input_ranges': [(-1.2, 0.5), (-0.07, 0.07)], # a vector of same length as 'dims' containing (min, max) tuples to rescale inputs
    'scale_output': True, # scales the output by number of active features
})

state = [-1.1, 0.03]
# returns an n-hot numpy array for active tiles
# if scale_output is true, then the "hot" tiles will be scaled by number of tilings
features = tc.encode(state)
print(features) # -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]

# returns the indices of the n-hot active tiles
# length of the list will be equal to number of tilings
# useful for using sparse vectors for faster computation, but harder to work with than "encode"
indices = tc.get_indices(state)
print(indices) # -> [4, 12, 31, 89]

num_features = tc.features()
print(num_features) # -> 16, the length of the vector generated by "encode"
```

### Options
 * `random_offset` - In most cases, having a uniform even offset is sufficient to have uncorrelated and high coverage of the statespace. Occasionally, it becomes important that the offsets are randomly generated to break any correlations or structure in the statespace.
 * `input_ranges` - It's important that the inputs are scaled between `[0, 1]` for this tile-coder to be most effective. When not scaled, we can potentially get into bad failure cases where only a small percentage of the tiles are ever active. This is true of all current tile-coder implementations. This implementation is robust to imperfect scaling by wrapping values larger than 1 back around to 0 (modular arithmetic), so as long as the units are _roughly_ correct then good performance can still be achieved.
 * `scale_output` - Has the `encode` function scale the outputs by the norm of the feature vector so that every feature vector has unit norm. This is simply a scalar multiple of every feature vector, which is often perfectly absorbed into the stepsize parameter, the effect of this option is to make it easier to find good stepsize by decoupling number of tilings from applicable stepsize range.
