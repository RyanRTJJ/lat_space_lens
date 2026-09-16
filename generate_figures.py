"""
Desc:       Regenerates the figures used in README.md.
            Run:  python generate_figures.py
Author:     ryan.rtjj@gmail.com
"""
from pathlib import Path

from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
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


# ---------------------------------------------------------------------------
# The distributed run: what is on disk, and what produces it.
#
# Drawn rather than photographed, so the numbers in it are the variables the
# code uses (N regions, S shards, C tasks per node, ...) and not one run's
# values. Colour carries the meaning:
#
#   MANILA      written by the driver before any worker starts (inputs)
#   KRAFT       written by a worker as it goes (progress: state and chunks)
#   BOOK_CLOTH  regions, the objects the whole pipeline moves around
#   IVORY       directories and other containers
#   CLOUD       machines
#   ERROR       a node dying, and the path back in
# ---------------------------------------------------------------------------
DARK_PAPER = '#fff9f2'
IVORY_LIGHT = '#fafaf7'
IVORY_MEDIUM = '#f0f0eb'
IVORY_DARK = '#e5e4df'
MANILA = '#ebdbbc'
KRAFT_LIGHT = '#FAEEE1'
KRAFT = '#d4a27f'
BOOK_CLOTH = '#cc785c'
ERROR = '#bf4d43'

CLOUD_LIGHT = '#bfbfba'
CLOUD_MEDIUM = '#91918d'
CLOUD_DARK = '#666663'
SLATE_LIGHT = '#40403E'
SLATE_MEDIUM = '#262625'
SLATE_DARK = '#191919'

# The diagram is laid out in its own units, 6 points to the unit, x to the right
# and y up. Everything in it is sized against the type rather than the other way
# round: DIAGRAM_MIN_FS is the floor, one line of it is DIAGRAM_MIN_FS / 6 units
# tall, and the boxes and gaps below are multiples of that.
DIAGRAM_POINTS_PER_UNIT = 6.0
DIAGRAM_MIN_FS = 10.0

DIAGRAM_W = 170
DIAGRAM_H = 196
DIAGRAM_FIGSIZE = (
    DIAGRAM_W * DIAGRAM_POINTS_PER_UNIT / 72,
    DIAGRAM_H * DIAGRAM_POINTS_PER_UNIT / 72,
)

FS_HEADLINE = 20.0
FS_PANEL_TITLE = 15.0
FS_DIR = 12.0               # a directory's own name
FS_FILE = 11.0              # a file name, and the labels in panel B
FS_ANNOT = DIAGRAM_MIN_FS   # the notes against a box's right edge, and captions


def _box(
        ax,
        x, y, w, h,
        fc=IVORY_LIGHT,
        ec=CLOUD_MEDIUM,
        label=None,
        sublabel=None,
        mono=False,
        fs=FS_FILE,
        lw=1.0,
        ls='-',
        text_color=SLATE_MEDIUM,
        label_pos='center',
        align='center',
        annot=None,
        annot_color=CLOUD_DARK,
        pad=1.0,
        zorder=2,
):
    """
    A rounded rectangle with a label, in diagram units. (x, y) is its bottom left.

    @param label_pos:   'center' puts the label in the middle of the box, 'top'
                        just inside the top edge, which is what the containers
                        (directories, the job, a node) use so their contents can
                        sit below it
    @param align:       'center' or 'left'. The file tree is left aligned, the
                        way a directory listing is
    @param annot:       a note against the box's right edge, on the label's line
    @return:            the (x, y, w, h) tuple, for the anchor helpers below
    """
    ax.add_patch(FancyBboxPatch(
        (x + pad / 2, y + pad / 2), w - pad, h - pad,
        boxstyle=f'round,pad={pad / 2},rounding_size=0.8',
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
    ))

    family = 'monospace' if mono else 'sans-serif'
    line = max(fs, DIAGRAM_MIN_FS) / DIAGRAM_POINTS_PER_UNIT
    if label_pos == 'top':
        text_y = y + h - line
    elif sublabel is not None:
        text_y = y + h / 2 + line * 0.6
    else:
        text_y = y + h / 2
    if label is not None:
        ax.text(
            x + 2.0 if align == 'left' else x + w / 2, text_y, label,
            ha='left' if align == 'left' else 'center', va='center',
            fontsize=max(fs, DIAGRAM_MIN_FS), family=family, color=text_color,
            zorder=zorder + 1,
        )
    if annot is not None:
        ax.text(
            x + w - 2.0, text_y, annot,
            ha='right', va='center', fontsize=FS_ANNOT, family=family,
            color=annot_color, zorder=zorder + 1,
        )
    if sublabel is not None:
        sub_y = text_y - line * 1.4
        ax.text(
            x + w / 2, sub_y, sublabel,
            ha='center', va='center', fontsize=FS_ANNOT, family='sans-serif',
            color=CLOUD_DARK, zorder=zorder + 1,
        )
    return (x, y, w, h)


def _top(box):
    x, y, w, h = box
    return (x + w / 2, y + h)


def _bottom(box):
    x, y, w, h = box
    return (x + w / 2, y)


def _left(box):
    x, y, w, h = box
    return (x, y + h / 2)


def _right(box):
    x, y, w, h = box
    return (x + w, y + h / 2)


def _arrow(
        ax,
        start,
        end,
        label=None,
        color=SLATE_LIGHT,
        ls='-',
        lw=1.2,
        fs=FS_ANNOT,
        label_offset=(0, 2.0),
        label_ha='center',
        connectionstyle='arc3,rad=0',
        zorder=4,
):
    """An annotated arrow between two points in diagram units."""
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle='-|>', mutation_scale=11, linewidth=lw, linestyle=ls,
        color=color, connectionstyle=connectionstyle,
        shrinkA=1.5, shrinkB=1.5, zorder=zorder,
    ))
    if label is not None:
        ax.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha=label_ha, va='center', fontsize=max(fs, DIAGRAM_MIN_FS),
            color=color, zorder=zorder + 1,
        )


def _caption(
        ax, x, y, text, fs=FS_ANNOT, color=CLOUD_DARK, ha='left', mono=False,
        mask=None,
):
    """
    @param mask:    a background colour to lay under the text, for the one caption
                    that sits on top of an arrow
    """
    ax.text(
        x, y, text, ha=ha, va='center', fontsize=max(fs, DIAGRAM_MIN_FS),
        color=color, family='monospace' if mono else 'sans-serif', zorder=6,
        linespacing=1.4,
        bbox=None if mask is None else {
            'facecolor': mask, 'edgecolor': 'none', 'pad': 2.0,
        },
    )


def _panel_title(ax, x, y, number, text):
    ax.text(
        x, y, f'{number}  {text}',
        ha='left', va='center', fontsize=FS_PANEL_TITLE, color=SLATE_DARK,
        weight='bold', zorder=6,
    )


# ---------------------------------------------------------------------------
# Panel A: the run directory, laid out the way a listing of it would be. One
# entry per line, left aligned, files before directories, since a file is one
# line tall and a directory is as tall as what is in it.
# ---------------------------------------------------------------------------
FS_ROW_H = 6.0              # a file: one line of type plus its padding
FS_NOTE_H = 5.0             # a '. . .' line, which has no box
FS_DIR_HEADER_H = 7.0       # the directory's own name, above its contents
FS_DIR_PAD = 2.0            # below the last entry, inside the directory
FS_GAP = 1.4                # between entries
FS_INDENT = 3.0             # how far a directory's contents sit inside it


def _fs_file(
        name, fc=MANILA, ec=KRAFT, annot=None,
        text_color=SLATE_MEDIUM, annot_color=CLOUD_DARK,
):
    return {
        'kind': 'file', 'name': name, 'fc': fc, 'ec': ec, 'annot': annot,
        'text_color': text_color, 'annot_color': annot_color,
    }


def _fs_note(text):
    return {'kind': 'note', 'name': text}


def _fs_dir(name, children=(), fc=IVORY_LIGHT, ec=CLOUD_LIGHT, annot=None, ls='-'):
    """
    Files first, whatever order they are passed in: a file is one line and a
    directory is a block, so listing the one-liners first keeps the names in a
    single column instead of scattering them down the side of the blocks.
    Everything else keeps the order it was given, so a trailing '. . .' line
    stays with the entry it continues.
    """
    children = list(children)
    children = (
        [child for child in children if child['kind'] == 'file']
        + [child for child in children if child['kind'] != 'file']
    )
    return {
        'kind': 'dir', 'name': name, 'children': children, 'fc': fc, 'ec': ec,
        'annot': annot, 'ls': ls,
    }


def _fs_height(node) -> float:
    if node['kind'] == 'file':
        return FS_ROW_H
    if node['kind'] == 'note':
        return FS_NOTE_H
    height = FS_DIR_HEADER_H + FS_DIR_PAD
    for child in node['children']:
        height += _fs_height(child) + FS_GAP
    return height


def _fs_draw(ax, node, x, y_top, width, depth=0):
    """Draw one entry with its top edge at y_top, and return the height it used."""
    height = _fs_height(node)

    if node['kind'] == 'note':
        _caption(ax, x + 2.0, y_top - height / 2, node['name'], mono=True)
        return height

    if node['kind'] == 'file':
        _box(
            ax, x, y_top - height, width, height,
            fc=node['fc'], ec=node['ec'], label=node['name'], annot=node['annot'],
            mono=True, fs=FS_FILE, align='left', text_color=node['text_color'],
            annot_color=node['annot_color'], zorder=2 + depth,
        )
        return height

    _box(
        ax, x, y_top - height, width, height,
        fc=node['fc'], ec=node['ec'], ls=node['ls'],
        label=node['name'], annot=node['annot'],
        mono=True, fs=FS_DIR, align='left', label_pos='top', zorder=2 + depth,
    )
    child_y = y_top - FS_DIR_HEADER_H
    for child in node['children']:
        child_y -= _fs_draw(
            ax, child, x + FS_INDENT, child_y, width - 2 * FS_INDENT, depth + 1,
        ) + FS_GAP
    return height


def _run_tree():
    """The whole of one run, as panel A draws it."""
    return _fs_dir(
        '<root>/',
        fc=IVORY_DARK, ec=CLOUD_MEDIUM,
        annot='a local dir, or gs://bucket/runs/run_001',
        children=[
            _fs_dir(
                'layer_000/',
                fc=IVORY_MEDIUM, ec=CLOUD_MEDIUM,
                annot='the first layer reversed',
                children=[
                    _fs_file('weights.pkl', annot="{'W', 'bias'}"),
                    _fs_file('layer.json', annot='shards, regions, weights hash'),
                    _fs_file(
                        'output.pkl', fc=BOOK_CLOTH, ec=CLOUD_DARK,
                        text_color=PAPER, annot_color=KRAFT_LIGHT,
                        annot='the next layer reads this',
                    ),
                    _fs_dir(
                        'inputs/',
                        annot='written by plan_layer',
                        children=[
                            _fs_file(
                                'shard_000000.pkl',
                                annot='(region_index, region) x R',
                            ),
                            _fs_file('shard_000001.pkl'),
                            _fs_note('. . .  S files, one per shard'),
                        ],
                    ),
                    _fs_dir(
                        'tasks/',
                        annot='written by the workers',
                        children=[
                            _fs_dir(
                                'shard_000000/',
                                fc=DARK_PAPER,
                                annot="task 0's progress",
                                children=[
                                    _fs_file(
                                        'state.json', fc=KRAFT, ec=CLOUD_DARK,
                                        text_color=SLATE_DARK, annot_color=SLATE_LIGHT,
                                        annot='region_i, num_chunks, done',
                                    ),
                                    _fs_file(
                                        'chunk_000000.pkl', fc=KRAFT_LIGHT, ec=KRAFT,
                                        annot="one window's finds",
                                    ),
                                    _fs_file(
                                        'chunk_000001.pkl', fc=KRAFT_LIGHT, ec=KRAFT,
                                    ),
                                    _fs_note('. . .  num_chunks in all'),
                                ],
                            ),
                            _fs_note('. . .  one directory per shard'),
                        ],
                    ),
                ],
            ),
            _fs_dir(
                'layer_001/,  layer_002/,  . . .',
                fc=IVORY_MEDIUM, ec=CLOUD_LIGHT, ls=(0, (3, 2)),
                annot='same shape',
            ),
        ],
    )


def _draw_filesystem_panel(ax):
    """
    The left panel: every file a run writes, nested and ordered as a listing of
    the run directory would be.

    The colours say who writes what: MANILA is the driver's, written before any
    worker starts; KRAFT is a worker's, written as it goes; BOOK_CLOTH is the
    regions the layer produces.
    """
    _panel_title(ax, 5, 177, 'A', 'What is in storage')
    _caption(
        ax, 5, 171,
        'one directory per layer, in the order the layers are reversed.\n'
        'The names are built in layout.py',
    )
    _fs_draw(ax, _run_tree(), x=5, y_top=166, width=68)


# ---------------------------------------------------------------------------
# Panel B: the path from one layer's regions to the next layer's.
# ---------------------------------------------------------------------------
def _draw_flow_panel(ax):
    """
    The right panel: the path from one layer's regions to the next layer's.

    Each step carries the module it lives in, since the point of the panel is to
    be able to go and read the code it describes. Boxes are kept to the height of
    the type in them, so that the space between them belongs to the arrows.
    """
    _panel_title(ax, 80, 177, 'B', 'How one layer is run')
    _caption(
        ax, 80, 171,
        'distributed.run(root, regions, layers, backend) does this for each\n'
        'layer, skipping whatever storage already has',
    )

    # --- the regions entering the layer ------------------------------------
    regions_in = [
        _box(
            ax, 80 + i * 11, 159, 9.5, 7,
            fc=BOOK_CLOTH, ec=CLOUD_DARK, label=f'r{i}', mono=True,
            text_color=PAPER,
        )
        for i in range(4)
    ]
    _caption(
        ax, 125, 162.5,
        '. . .  N regions in\nlayer 0: the starting region,\nthen output.pkl of the layer before',
    )

    # --- plan_layer ---------------------------------------------------------
    plan = _box(
        ax, 80, 139, 50, 9,
        fc=IVORY_DARK, ec=CLOUD_DARK,
        label='plan_layer', sublabel='distributed.py, before any task runs',
    )
    _arrow(
        ax, _bottom(regions_in[1]), _top(plan),
        'shuffle(seed),\ndeal round-robin',
        label_offset=(2.5, 0.0), label_ha='left',
    )
    _caption(ax, 134, 145, 'S = num_shards,\ndefault 4 x slots', mono=True)
    _caption(ax, 134, 137, 'R = N / S regions\nper shard', mono=True)

    # --- the shards ---------------------------------------------------------
    shards = [
        _box(
            ax, 80 + i * 18, 119.5, 16, 8.5,
            fc=MANILA, ec=KRAFT, label=f'shard {i}', sublabel='R regions',
        )
        for i in range(3)
    ]
    _caption(ax, 136, 123.5, '. . .  S shards')
    _arrow(ax, _bottom(plan), _top(shards[0]), None)

    # --- the job, its nodes, and their task slots ---------------------------
    job = _box(
        ax, 80, 78, 66, 30,
        fc=IVORY_MEDIUM, ec=CLOUD_MEDIUM,
        label='backend.run_tasks: one task per unfinished shard',
        sublabel='distributed.py, one process per shard',
        label_pos='top',
    )
    _caption(ax, 148, 104, 'task i takes\nshard i, from\nBATCH_TASK_INDEX')
    _caption(ax, 148, 92, 'Cloud Batch:\ntaskCount = S,\nSpot VMs, retry\non exit 50001')
    _caption(ax, 148, 82, 'C = tasks\nper node', mono=True)

    tasks = []
    for n in range(2):
        node = _box(
            ax, 83 + n * 31.5, 80, 28.5, 20,
            fc=CLOUD_LIGHT, ec=CLOUD_DARK, label=f'node {n}', mono=True,
            label_pos='top',
        )
        for c in range(2):
            tasks.append(_box(
                ax, node[0] + 2.5 + c * 12.5, 82.5, 11.5, 9.5,
                fc=IVORY_LIGHT, ec=CLOUD_DARK,
                label=f'task {2 * n + c}', sublabel='worker', mono=True,
            ))

    for shard, task in zip(shards[:2], tasks[:2]):
        _arrow(ax, _bottom(shard), _top(task), None, color=CLOUD_DARK)
    _arrow(
        ax, _bottom(shards[2]), _top(tasks[2]), None, color=CLOUD_DARK,
        connectionstyle='arc3,rad=-0.12',
    )

    # A node lost partway through, which is what all the checkpointing is for.
    dead = tasks[3]
    ax.add_patch(FancyBboxPatch(
        (dead[0] + 0.5, dead[1] + 0.5), dead[2] - 1.0, dead[3] - 1.0,
        boxstyle='round,pad=0.6,rounding_size=0.8',
        facecolor='none', edgecolor=ERROR, linewidth=1.8, zorder=6,
    ))
    _caption(ax, dead[0] + dead[2] / 2, 81.2, 'preempted', color=ERROR, ha='center')

    # --- one task's inner loop ---------------------------------------------
    loop = _box(
        ax, 80, 30, 66, 36,
        fc=DARK_PAPER, ec=CLOUD_MEDIUM,
        label='inside task i: its R regions, one search each',
        sublabel='worker.py, one process per shard',
        label_pos='top',
    )
    _arrow(ax, _bottom(tasks[0]), (92, _top(loop)[1]), None, color=CLOUD_DARK)
    _caption(
        ax, 148, 56,
        'one region at\na time, in\nwindows of\ncheckpoint_\nseconds, until\nthe region\nis done',
    )

    windows = []
    for i in range(3):
        x = 84 + i * 20
        windows.append(_box(
            ax, x, 52, 18, 8,
            fc=IVORY_LIGHT, ec=CLOUD_DARK,
            label=f'window {i}', sublabel='60 s' if i == 0 else None,
        ))
        if i == 1:
            _caption(
                ax, x + 9, 45.5, 'found none:\nno chunk', ha='center',
                mask=DARK_PAPER,
            )
            continue
        chunk = _box(
            ax, x, 41, 18, 7,
            fc=KRAFT_LIGHT, ec=KRAFT, label='chunk', mono=True,
        )
        _arrow(ax, _bottom(windows[-1]), _top(chunk), None, color=KRAFT)
    for left_window, right_window in zip(windows, windows[1:]):
        _arrow(ax, _right(left_window), _left(right_window), None, color=CLOUD_DARK)

    state = _box(
        ax, 84, 32, 56, 7,
        fc=KRAFT, ec=CLOUD_DARK,
        label='state.json    rewritten every window, chunk first',
        mono=True, text_color=SLATE_DARK,
    )
    for i, window in enumerate(windows):
        _arrow(
            ax, (window[0] + window[2] / 2, 52 if i == 1 else 41),
            (window[0] + window[2] / 2, _top(state)[1]),
            None, color=KRAFT, lw=1.0, ls=(0, (2, 2)),
        )

    # --- gather, and the layer's output ------------------------------------
    gather = _box(
        ax, 80, 14, 56, 9,
        fc=IVORY_DARK, ec=CLOUD_DARK,
        label='gather_layer + flatten_layer(substitute)',
        sublabel='distributed.py, once every shard says done',
    )
    _arrow(
        ax, (92, loop[1]), (92, _top(gather)[1]),
        "every shard's chunks,\nmerged by region_i",
        label_offset=(3.0, 0.0), label_ha='left',
    )
    regions_out = [
        _box(
            ax, 140 + i * 11, 15, 9.5, 7,
            fc=BOOK_CLOTH, ec=CLOUD_DARK, label=f"r{i}'", mono=True,
            text_color=PAPER,
        )
        for i in range(2)
    ]
    _arrow(ax, _right(gather), _left(regions_out[0]), None)
    _caption(ax, 140, 26, ". . .  N' regions, into output.pkl")

    # The resume path: the preempted task is run again and picks up where its
    # state.json says, not at the start of its shard.
    _arrow(
        ax, (dead[0] + dead[2] / 2, dead[1]), (_right(state)[0], _right(state)[1]),
        'the rerun reads\nstate.json and\nresumes mid-search',
        color=ERROR, ls=(0, (4, 2)),
        connectionstyle='arc3,rad=-0.3',
        label_offset=(12.0, 12.0), label_ha='center',
    )


def _draw_legend(ax, x0, y0, width):
    """What each colour means, since the colours are the only key to the panels."""
    entries = [
        (MANILA, KRAFT, 'driver writes'),
        (KRAFT, CLOUD_DARK, 'worker writes'),
        (BOOK_CLOTH, CLOUD_DARK, 'regions'),
        (IVORY_MEDIUM, CLOUD_MEDIUM, 'directories, steps'),
        (CLOUD_LIGHT, CLOUD_DARK, 'machines, slots'),
        ('none', ERROR, 'node lost, way back'),
    ]
    swatch = 5.0
    step = width / len(entries)
    for i, (fc, ec, text) in enumerate(entries):
        x = x0 + i * step
        y = y0
        ax.add_patch(FancyBboxPatch(
            (x, y), swatch, swatch * 0.75,
            boxstyle='round,pad=0.4,rounding_size=0.6',
            facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3,
        ))
        _caption(ax, x + swatch + 2.5, y + swatch * 0.375, text)


def fig_distributed_architecture():
    """
    The storage layout and the run that fills it, side by side, as an SVG so it
    stays sharp in the README at any width.
    """
    fig, ax = plt.subplots(figsize=DIAGRAM_FIGSIZE)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, DIAGRAM_W)
    ax.set_ylim(0, DIAGRAM_H)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax.text(
        5, DIAGRAM_H - 7, 'A distributed reversal: one layer at a time',
        ha='left', va='center', fontsize=FS_HEADLINE, color=SLATE_DARK,
        weight='bold',
    )
    _caption(
        ax, 5, DIAGRAM_H - 14,
        'Nothing is kept in memory between the steps below: every arrow crosses '
        'storage,\nwhich is what lets any worker, or the driver, die and be run again.',
        fs=12, color=CLOUD_DARK,
    )

    _draw_filesystem_panel(ax)
    _draw_flow_panel(ax)
    _draw_legend(ax, x0=6, y0=5, width=162)

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(OUT_DIR / 'distributed_architecture.svg', format='svg')
    plt.close(fig)

if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': FS})
    fig_orthant_preimage()
    fig_probe_constraint()
    fig_probe_preimage()
    fig_distributed_architecture()
    print(f'wrote figures to {OUT_DIR}')
