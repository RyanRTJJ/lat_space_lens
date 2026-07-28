# lat_space_lens

Backpropping feature constraints through ReLU networks to input spaces.

## Install

From GitHub:

```bash
pip install git+https://github.com/RyanRTJJ/lat_space_lens.git
```

Locally, for development:

```bash
git clone https://github.com/RyanRTJJ/lat_space_lens.git
cd lat_space_lens
pip install -e .
```

## Updating

A `git+` install snapshots the code at install time; it does not track the
repo. To pick up newer commits:

```bash
pip install --upgrade --force-reinstall git+https://github.com/RyanRTJJ/lat_space_lens.git
```

`--force-reinstall` is needed because pip will otherwise consider the
requirement already satisfied whenever the version string hasn't changed.

For reproducible installs, pin to a tag (or a commit hash) rather than
tracking the default branch:

```bash
pip install "git+https://github.com/RyanRTJJ/lat_space_lens.git@v0.1.0"
```

## Use

```python
import numpy as np
from lat_space_lens import ConstraintSet, plot_region, reverse_relu_helper

# "feature 0 of the final layer is active"
end_constraint = ConstraintSet(
    np.eye(d_out)[0],
    1e-4,
    ConstraintSet.GEQ,
)

# Walk backwards through the layers
Z_to_region = end_constraint.reverse_relu(W, b)
regions = ConstraintSet.flatten_region_dicts(Z_to_region)
regions = [r.reverse_up_proj(W, b) for r in regions]

# If the input space is 2D:
for region in regions:
    plot_region(ax, region, plot_z=5.0, color='coral', alpha=0.8)
```

`reverse_relu_helper(region, W, b)` is the top-level (picklable) form of
`reverse_relu` + flatten, for use with `ProcessPoolExecutor.map`.
