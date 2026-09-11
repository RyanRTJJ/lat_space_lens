"""
Desc:       Regenerates the figures used in README.md.
            Run:  python generate_figures.py
Author:     ryan.rtjj@gmail.com
"""
from pathlib import Path

from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from lat_space_lens import ConstraintSet, plot_region
from lat_space_lens.plotting import _convex_polygon_vertices

PAPER = '#fcfbf8'
CMAP = 'coolwarm'
FS = 9
TOL = 1e-9

OUT_DIR = Path(__file__).parent / 'figures'

# The example layer: p = U @ x + b_U, z = ReLU(p). x is 2D, z is 3D.
U = np.array([
    [1.0,  0.0],
    [-0.5,  np.sqrt(3) / 2],
    [-0.5, -np.sqrt(3) / 2],
])
B_U = -0.3 * np.ones(3)

# The probe we constrain in post-ReLU space: probe @ z > 0.3
PROBE = np.array([1 / 4, 1 / 4, -np.sqrt(3) / 4])
PROBE_B = 0.3

CUBE_L = 3.0        # the positive hypercube is [0, CUBE_L]^3
PLANE_L = 2.7       # the plane is drawn over x in [-PLANE_L, PLANE_L]^2
PAD = 0.1          # gap left at the origin so highlights do not overlap
AXIS_L = 4.6
PLOT_L = 4.0
VIEW = dict(elev=25, azim=-45, roll=0)

# Fill opacities. The unhighlighted hull goes down first and everything else is
# drawn over it, so it has to fade as the figure gets busier. The wedge needs to
# go fainter than the cube because more of its faces stack up along the line of
# sight, and matplotlib composites every one of them.
HULL_ALPHA_SOLO = 0.85      # nothing on top: the hull is the subject
HULL_ALPHA_UNDER_CUBE = 0.55
HULL_ALPHA_UNDER_WEDGE = 0.32
HIGHLIGHT_ALPHA = 0.8       # the coloured faces / edges / origin
PLANE_ALPHA = 0.1           # the grey backdrop the pre-images sit on

# Axes3D recomputes the zorder of every Collection from its depth, so the hull's
# faces land somewhere in a band a little above the axis zorder (1.5). Line3D is
# the one artist it leaves alone, so the edges and the origin only have to clear
# that band -- hence numbers this far out.
ZORDER_EDGE = 100
ZORDER_ORIGIN = 101

# 7 origin-connected faces of the positive orthant, ordered by dimension.
# (The 8th, the interior, keeps the default colour.)
MANIFOLDS = [
    (0, 1),
    (0,),
    (0, 2),
    (0, 1, 2),
    (2,),
    (1, 2),
    (1,),
]
# MANIFOLDS = [
#     (0, 1, 2),                          # the origin
#     (1, 2), (0, 2), (0, 1),             # the three edges
#     (2,), (1,), (0,),                   # the three faces
# ]
_CMAP9 = plt.get_cmap(CMAP, 9)
DEFAULT_COLOR = _CMAP9(4)
COLORS = {Z: _CMAP9(i) for Z, i in zip(MANIFOLDS, [0, 1, 2, 3, 6, 7, 8])}


class Arrow3D(FancyArrowPatch):
    """
    A helper class to draw 3D arrows.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def beautify_ax_3d(ax: plt.Axes, z: float):
    """
    @param z:               limits of plot
    """
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-z, z)
    ax.set_ylim(-z, z)
    ax.set_zlim(-z, z)
    ax.set_facecolor(PAPER)
    ax.grid(False)
    ax.set_axis_off()


def add_axis_lines_3d(ax, z: float):
    arrow_kwargs = {
        'arrowstyle': '-|>', 'mutation_scale': 10, 'lw': 1, 'color': 'black',
    }
    ax.add_artist(Arrow3D([-z, z], [0, 0], [0, 0], **arrow_kwargs))
    ax.add_artist(Arrow3D([0, 0], [-z, z], [0, 0], **arrow_kwargs))
    ax.add_artist(Arrow3D([0, 0], [0, 0], [-z, z], **arrow_kwargs))


# ---------------------------------------------------------------------------
# Post-ReLU side: the pieces of a polyhedron that sit on each face of the
# positive orthant.
# ---------------------------------------------------------------------------
def orthant_box(extra_A=None, extra_b=None):
    """
    Half-spaces A @ z >= b cutting out [0, CUBE_L]^3, optionally intersected
    with extra constraints.
    """
    A = np.vstack([np.eye(3), -np.eye(3)])
    b = np.concatenate([np.zeros(3), -CUBE_L * np.ones(3)])
    if extra_A is not None:
        A = np.vstack([A, extra_A])
        b = np.concatenate([b, np.atleast_1d(extra_b)])
    return A, b


def manifold_piece(A: np.ndarray, b: np.ndarray, Z: tuple):
    """
    The slice of {z : A @ z >= b} lying on the orthant face that is zero on
    the dims in `Z`, shrunk by PAD along every dim that is free. Returns a
    (n, 3) array of vertices (n == 1 point, n == 2 segment, n > 2 polygon),
    or None if the slice is empty.
    """
    free = [i for i in range(3) if i not in Z]

    # Substitute z_j = 0 for j in Z: those columns just drop out.
    A_r = A[:, free]
    b_r = b.copy()

    # The padding is one more half-space per free dim.
    if free:
        A_r = np.vstack([A_r, np.eye(len(free))])
        b_r = np.concatenate([b_r, PAD * np.ones(len(free))])

    def lift(coords_2d):
        out = np.zeros((len(coords_2d), 3))
        out[:, free] = coords_2d
        return out

    if len(free) == 0:
        return np.zeros((1, 3)) if np.all(b <= TOL) else None

    if len(free) == 1:
        a = A_r[:, 0]
        lo, hi = -np.inf, np.inf
        for a_k, b_k in zip(a, b_r):
            if abs(a_k) < TOL:
                if b_k > TOL:
                    return None
            elif a_k > 0:
                lo = max(lo, b_k / a_k)
            else:
                hi = min(hi, b_k / a_k)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < TOL:
            return None
        return lift(np.array([[lo], [hi]]))

    vertices = _convex_polygon_vertices(A_r, b_r, TOL)
    if vertices is None:
        return None
    return lift(vertices)


def draw_manifold_piece(ax, piece: np.ndarray, color, custom_zorder = None):
    """
    A point is scattered, a segment is a thick line, a polygon is filled.
    """
    if len(piece) == 1:
        # ax.plot rather than ax.scatter: a scatter is a Collection, so Axes3D
        # would overwrite the zorder and bury the origin under the hull.
        ax.plot(
            *[[c] for c in piece[0]],
            color=color, marker='o', markersize=7, linestyle='',
            zorder=ZORDER_ORIGIN,
        )
    elif len(piece) == 2:
        ax.plot(
            *piece.T, color=color, lw=4, solid_capstyle='round',
            zorder=custom_zorder if custom_zorder is not None else ZORDER_EDGE,
        )
    else:
        ax.add_collection3d(Poly3DCollection(
            [piece], facecolor=color, edgecolor='white', linewidth=1,
            alpha=HIGHLIGHT_ALPHA,
        ))


# ---------------------------------------------------------------------------
# Pre-ReLU side: the pre-image regions, drawn on the plane {U @ x + b_U}.
# ---------------------------------------------------------------------------
def plane_vertices():
    corners = np.array([
        [PLANE_L, PLANE_L], [PLANE_L, -PLANE_L],
        [-PLANE_L, -PLANE_L], [-PLANE_L, PLANE_L],
    ])
    return corners @ U.T + B_U


def preimage_polygons(constraint_set: ConstraintSet):
    """
    Reverses `constraint_set` through the ReLU and the up-projection, and
    returns {Z tuple: (n, 3) vertices of that region lifted onto the plane}.
    """
    Z_to_region = constraint_set.reverse_relu(U, B_U)

    out = {}
    for Z, region in Z_to_region.items():
        x_region = region.reverse_up_proj(U, B_U)

        # Clip to the drawn part of the plane, then read off the polygon.
        A = np.vstack([x_region.A, np.eye(2), -np.eye(2)])
        b = np.concatenate([
            x_region.b, -PLANE_L * np.ones(2), -PLANE_L * np.ones(2),
        ])
        vertices = _convex_polygon_vertices(A, b, TOL)
        if vertices is None:
            continue
        out[Z] = vertices @ U.T + B_U
    return out


def draw_plane_and_preimages(ax, polygons: dict):
    ax.add_collection3d(Poly3DCollection(
        [plane_vertices()], facecolor='gray', alpha=PLANE_ALPHA,
    ))
    for Z, vertices in polygons.items():
        ax.add_collection3d(Poly3DCollection(
            [vertices],
            facecolor=COLORS[Z],
            edgecolor='white',
            linewidth=1,
            alpha=HIGHLIGHT_ALPHA,
        ))


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------
def fig_orthant_preimage():
    """
    The 7 origin-connected faces of the positive hypercube, next to the
    pre-image of each one on the plane p = U @ x + b_U.
    """
    fig, ax = plt.subplots(
        1, 1, figsize=(7, 7), subplot_kw={'projection': '3d'},
    )

    # The hypercube itself, unhighlighted.
    cube = ConstraintSet(np.eye(3), np.zeros(3), [ConstraintSet.GEQ] * 3)
    plot_region(
        ax, cube, plot_z=CUBE_L,
        color=DEFAULT_COLOR, alpha=HULL_ALPHA_UNDER_CUBE,
    )

    A, b = orthant_box()
    for Z in MANIFOLDS:
        piece = manifold_piece(A, b, Z)
        if piece is not None:
            if Z == (0, 2):
                draw_manifold_piece(ax, piece, COLORS[Z], custom_zorder=0)
            else:
                draw_manifold_piece(ax, piece, COLORS[Z])

    draw_plane_and_preimages(ax, preimage_polygons(cube))

    beautify_ax_3d(ax, z=PLOT_L)
    add_axis_lines_3d(ax, z=AXIS_L)
    ax.view_init(**VIEW)
    fig.subplots_adjust(left=-0.06, right=1.06, bottom=-0.06, top=1.06)
    fig.savefig(OUT_DIR / 'orthant_preimage.png', dpi=200)
    plt.close(fig)


def fig_probe_constraint():
    """
    Left: the positive hypercube. Right: the same cube intersected with the
    probe's half-space.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5.5), subplot_kw={'projection': '3d'},
    )

    cube = ConstraintSet(np.eye(3), np.zeros(3), [ConstraintSet.GEQ] * 3)
    constrained = ConstraintSet(
        np.vstack([np.eye(3), PROBE]),
        np.concatenate([np.zeros(3), [PROBE_B]]),
        [ConstraintSet.GEQ] * 4,
    )

    titles = ['the positive orthant', 'intersected with the probe half-space']
    for i, (ax, region, title) in enumerate(zip(axes, [cube, constrained], titles)):
        plot_region(
            ax, region, plot_z=CUBE_L,
            color=DEFAULT_COLOR, alpha=HULL_ALPHA_SOLO,
        )
        beautify_ax_3d(ax, z=PLOT_L)
        add_axis_lines_3d(ax, z=AXIS_L)
        ax.view_init(**VIEW)
        fig.text(0.25 + 0.5 * i, 0.1, title, ha='center', fontsize=FS)

    fig.subplots_adjust(left=-0.06, right=1.06, bottom=-0.10, top=1.04, wspace=-0.05)
    fig.savefig(OUT_DIR / 'probe_constraint.png', dpi=200)
    plt.close(fig)


def fig_probe_preimage():
    """
    The probe-constrained polyhedron, and its pre-image on the plane, face by
    face, in matching colours.
    """
    fig, ax = plt.subplots(
        1, 1, figsize=(7, 7), subplot_kw={'projection': '3d'},
    )

    constrained = ConstraintSet(
        np.vstack([np.eye(3), PROBE]),
        np.concatenate([np.zeros(3), [PROBE_B]]),
        [ConstraintSet.GEQ] * 4,
    )
    plot_region(
        ax, constrained, plot_z=CUBE_L,
        color=DEFAULT_COLOR, alpha=HULL_ALPHA_UNDER_WEDGE,
    )

    A, b = orthant_box(PROBE, PROBE_B)
    for Z in MANIFOLDS:
        piece = manifold_piece(A, b, Z)
        if piece is not None:
            draw_manifold_piece(ax, piece, COLORS[Z])

    draw_plane_and_preimages(ax, preimage_polygons(constrained))

    beautify_ax_3d(ax, z=PLOT_L)
    add_axis_lines_3d(ax, z=AXIS_L)
    ax.view_init(**VIEW)
    fig.subplots_adjust(left=-0.06, right=1.06, bottom=-0.06, top=1.06)
    fig.savefig(OUT_DIR / 'probe_preimage.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': FS})
    fig_orthant_preimage()
    fig_probe_constraint()
    fig_probe_preimage()
    print(f'wrote figures to {OUT_DIR}')
