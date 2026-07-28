# lat_space_lens

Backpropping feature constraints through ReLU networks to input spaces.

## Install

From GitHub:

```bash
pip install git+https://github.com/<your-username>/lat_space_lens.git
```

Locally, for development:

```bash
pip install -e /Users/ryan.tan/Documents/lat_space_lens
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
