"""
Desc:       2D plotting of ConstraintSet regions.
Author:     ryan.rtjj@gmail.com
"""
from itertools import combinations

import numpy as np

from .constraints import ConstraintSet


def plot_region(
        ax,
        region: ConstraintSet,
        plot_z: int,
        color,
        alpha: float):
    """
    If your d_model happens to be 2, you can use this.
    """
    TOL = 1e-9
    num_constraints, d = region.A.shape
    assert d == 2, \
        f'plot_region only implemented for 2D regions, got shape: {region.A.shape}'

    # Add box constraints
    A_full = np.vstack([
        region.A,
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ])
    b_full = np.concatenate([
        region.b,
        [-plot_z, -plot_z, -plot_z, -plot_z]
    ])

    # Find all intersections (there's like num_constraints ** 2 of them)
    vertices = []
    n = len(A_full)

    for i, j in combinations(range(n), 2):
        # Solve for point of intersection
        A_pair = A_full[[i, j]]
        b_pair = b_full[[i, j]]

        try:
            x = np.linalg.solve(A_pair, b_pair)
            # Does this satisfy all other constraints?
            if np.all(A_full @ x >= b_full - TOL):
                vertices.append(x)
        except np.linalg.LinAlgError:
            # parallel, continue
            continue

    if not vertices:
        return

    vertices = np.array(vertices)

    # Remove duplicates
    vertices = np.unique(vertices.round(decimals=10), axis=0)

    # Order vertices by angle from centroid
    centroid = vertices.mean(axis=0)
    angles = np.arctan2(
        vertices[:, 1] - centroid[1],
        vertices[:, 0] - centroid[0],
    )
    vertices = vertices[np.argsort(angles)]

    # Plot
    vertices_closed = np.vstack([vertices, vertices[0]])
    ax.fill(vertices_closed[:, 0], vertices_closed[:, 1], color=color, alpha=alpha)
