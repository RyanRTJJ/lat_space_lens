"""
Desc:       2D/3D plotting of ConstraintSet regions.
Author:     ryan.rtjj@gmail.com
"""
from itertools import combinations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .constraints import ConstraintSet


def _convex_polygon_vertices(A_full: np.ndarray, b_full: np.ndarray, tol: float) -> np.ndarray | None:
    """
    Given 2D half-plane constraints A_full @ x >= b_full, returns the
    vertices of the resulting convex polygon, ordered around the
    perimeter, by finding every pairwise intersection of constraint
    boundary lines that satisfies all other constraints.

    @return:    (n, 2) array of ordered vertices, or None if the region
                has fewer than 3 vertices (empty / degenerate).
    """
    vertices = []
    n = len(A_full)

    for i, j in combinations(range(n), 2):
        # Solve for point of intersection
        A_pair = A_full[[i, j]]
        b_pair = b_full[[i, j]]

        try:
            x = np.linalg.solve(A_pair, b_pair)
            # Does this satisfy all other constraints?
            if np.all(A_full @ x >= b_full - tol):
                vertices.append(x)
        except np.linalg.LinAlgError:
            # parallel, continue
            continue

    if not vertices:
        return None

    vertices = np.array(vertices)

    # Remove duplicates
    vertices = np.unique(vertices.round(decimals=10), axis=0)
    if len(vertices) < 3:
        return None

    # Order vertices by angle from centroid
    centroid = vertices.mean(axis=0)
    angles = np.arctan2(
        vertices[:, 1] - centroid[1],
        vertices[:, 0] - centroid[0],
    )
    return vertices[np.argsort(angles)]


def _orthonormal_complement(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a 3D vector, returns 2 orthonormal vectors (u, v) that span its
    orthogonal complement (i.e. the plane normal to `normal`).
    """
    n_hat = normal / np.linalg.norm(normal)

    # Pick a helper vector guaranteed not to be (near-)parallel to n_hat
    helper = np.array([1.0, 0.0, 0.0])
    if abs(n_hat @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    u = helper - (helper @ n_hat) * n_hat
    u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    return u, v


def _plot_region_2d(ax, region: ConstraintSet, plot_z: int, color, alpha: float):
    TOL = 1e-9

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

    vertices = _convex_polygon_vertices(A_full, b_full, TOL)
    if vertices is None:
        return

    # Plot
    vertices_closed = np.vstack([vertices, vertices[0]])
    ax.fill(vertices_closed[:, 0], vertices_closed[:, 1], color=color, alpha=alpha)


def _plot_region_3d(
        ax,
        region: ConstraintSet,
        plot_z: int,
        color,
        alpha: float,
):
    """
    Plots the polyhedron region.A @ x >= region.b intersected with the
    [-plot_z, plot_z]^3 cube, face by face (`ax` must have
    projection='3d'). Each candidate face is the boundary PLANE of either
    one of region's own linear inequalities, or one of the cube's own 6
    faces (rank-2 surfaces in 3D, same as the box constraints' boundary
    lines were rank-1 in 2D).

    For each candidate face plane, we project every OTHER constraint onto
    that plane's local 2D coordinates (u, v) -- collapsing it to exactly
    the 2D half-plane-intersection problem `_plot_region_2d` already
    solves -- reuse `_convex_polygon_vertices` to find that face's
    polygon, then lift its vertices back into 3D.
    """
    TOL = 1e-9

    # Add box constraints (the cube's 6 faces)
    A_full = np.vstack([
        region.A,
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ])
    b_full = np.concatenate([
        region.b,
        [-plot_z, -plot_z, -plot_z, -plot_z, -plot_z, -plot_z]
    ])

    n = len(A_full)
    for i in range(n):
        normal = A_full[i]
        offset = b_full[i]
        norm_sq = normal @ normal
        if norm_sq < TOL:
            continue

        # A particular point on this constraint's boundary plane
        p0 = normal * (offset / norm_sq)
        u, v = _orthonormal_complement(normal)

        # Project every OTHER constraint onto this plane's local (s, t) coords:
        # substituting x = p0 + s*u + t*v into A_j @ x >= b_j gives
        # [A_j @ u, A_j @ v] @ [s, t] >= b_j - A_j @ p0
        other_idxs = [j for j in range(n) if j != i]
        A_other = A_full[other_idxs]
        b_other = b_full[other_idxs]

        A_2d = np.stack([A_other @ u, A_other @ v], axis=1)
        b_2d = b_other - A_other @ p0

        face_vertices_2d = _convex_polygon_vertices(A_2d, b_2d, TOL)
        if face_vertices_2d is None:
            continue

        face_vertices_3d = (
            p0[None, :]
            + face_vertices_2d[:, [0]] * u[None, :]
            + face_vertices_2d[:, [1]] * v[None, :]
        )

        polygon = Poly3DCollection(
            [face_vertices_3d],
            facecolor=color,
            alpha=alpha,
            edgecolor='white',
            linewidth=1,
        )
        ax.add_collection3d(polygon)


def plot_region(
        ax,
        region: ConstraintSet,
        plot_z: int,
        color,
        alpha: float):
    """
    Plots `region` (intersected with a plot_z-radius square / cube) onto
    `ax`. Supports region.A.shape[1] (d) in (2, 3): d == 2 fills a polygon
    on a standard Axes; d == 3 renders the polyhedron face-by-face as
    Poly3DCollection patches (ax must have projection='3d').
    """
    num_constraints, d = region.A.shape
    assert d in (2, 3), \
        f'plot_region only implemented for 2D/3D regions, got shape: {region.A.shape}'

    if d == 2:
        _plot_region_2d(ax, region, plot_z, color, alpha)
    else:
        _plot_region_3d(ax, region, plot_z, color, alpha)
