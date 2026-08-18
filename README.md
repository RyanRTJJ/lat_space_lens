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

## The Math

This section explains `reverse_add_up_proj` and `reverse_up_proj` (complicated). Suppose we have:

$$
\begin{align*}
p &= Wx + My + \mathrm{bias},\\
z &= \mathrm{ReLU}(p)
\end{align*}
$$

Further suppose that $x$ is the product of earlier network layers, but $y$ is a fresh input (such as in the case of an RNN), we want to reverse a constraint on $z$ in the form $Az > b$ to constraints on $y$ and $x$ (or further upstream inputs that made up $x$).

### `reverse_add_up_proj`

We have to think of:
$$
\begin{align*}
Az &> b
\end{align*}
$$

as:

$$
\begin{align*}
A \cdot \text{ReLU}\left( \begin{bmatrix} W \mid M \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \mathrm{bias} \right) &> b
\end{align*}
$$


**Block structure.** 

It is important to track $\begin{bmatrix} W \mid M \end{bmatrix}$ explicitly as 2 blocks instead of 1 homogenous vector. This is because in something like an RNN, $y$ is a freshly injected input while $x$ comes from previous RNN layers. This means that the $x$ part has to continue reversing while the $y$ part is done.

So, in `ConstraintSet`, there is a member called `.block_sizes: list[int]` that is literally tracking the sizes of these blocks.

With the `ConstraintSet` describing $Ap \geq b$, we have:

```
block_sizes = [A.shape[1]]
```

After `reverse_add_up_proj`, each of the regions output by `reverse_add_up_proj` hence have:

```
block_sizes = [W.shape[1], M.shape[1]]
```

The leading (left-most) block is always assumed to continue reversing into previous layers, while everything else (called `suffix`) is assumed to be injected inputs (from here and downstream).

`reverse_add_up_proj` is simply a thin wrapper around `reverse_up_proj` that book-keeps `block_sizes`.

### `reverse_up_proj`

Once a layer has injected a fresh input, the variable being constrained is no
longer one homogeneous vector. It is a concatenation

$$v = \begin{bmatrix} x \\ y \end{bmatrix}$$

where $x$ is the **active** block, the only part still being reversed, and $y$ is the **suffix**: injected inputs that are already done and just riding along.

As mentioned, only $x$ is the output of previous layers, so $x$ is the result of some previous up-projection:

$$
x = Wu + \mathrm{bias}
$$

The layer we are reversing maps into the active block only, with $u$ the new active variable, while $y$ passes through untouched. So the map on the *whole* variable is $W$ in one corner and an identity in the other:

$$\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} W & 0 \\ 0 & I \end{bmatrix} \begin{bmatrix} u \\ y \end{bmatrix} + \begin{bmatrix} \mathrm{bias} \\ 0 \end{bmatrix}$$

That block-diagonal matrix $W_{\mathrm{full}}$ is exactly what `_carry_suffix` builds:

```python
W_full    = block_diag(W, np.eye(d_suffix))
bias_full = np.concatenate([bias, np.zeros(d_suffix)])
```

Our goal is to now back out constraints on $\begin{bmatrix} u \\ y \end{bmatrix}$. So from:

$$
\begin{align*}
A \left( W_{\mathrm{full}} \begin{bmatrix} u \\ y \end{bmatrix} + \mathrm{bias}_\mathrm{full} \right) &> b
\end{align*}
$$

We get:

$$
\begin{align*}
A \, W_{\mathrm{full}}  \begin{bmatrix} u \\ y \end{bmatrix} &> b - A\,\mathrm{bias}_{\mathrm{full}}
\end{align*}
$$


```python
new_A = self.A @ W_full
new_b = self.b - self.A @ bias_full
```

and the bookkeeping is just the active block swapping its width:

```python
block_sizes = [W.shape[1]] + self.block_sizes[1:]
```

**With no suffix, none of this applies.** `_carry_suffix` short-circuits when
`d_suffix == 0` and hands back `W` and `bias` untouched, so an un-blocked
constraint set behaves exactly as it did before `block_sizes` existed.

`reverse_down_proj` uses the same helper. Nothing above assumes $W$ is tall.
