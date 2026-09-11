"""Test cases for the reverse-ReLU activation pattern search.

The weights and the linear probe live in constants.py, which stores them
untruncated so that this file depends on neither the alg_zoo library nor network
access. Each test truncates them itself.

M_16_10_* is the hidden_size=16, seq_len=10 model of the blog post, from
example_2nd_argmax(), and M_16_10_PROBE its relu_1_post probe.
"""
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import constants
from constants import M_16_10_PROBE
from constants import M_16_10_W_hh
from constants import M_16_10_W_hi
from lat_space_lens import ConstraintSet
from lat_space_lens import flatten_layer
from lat_space_lens import reverse_relu_layer
from lat_space_lens import resolve_executor

# figures_16_10.py substitutes this for the probe threshold, because the probe is
# fitted with a hard-coded intercept of 0.0 and the region would otherwise be
# degenerate.
TIIINY = 0.0001

# The palette of algozoo/figures_16_10.py, so that a figure produced here sits
# alongside the ones produced there without a change of colour scheme.
PAPER = '#fcfbf8'
CLOUD_LIGHT = '#bfbfba'
CLOUD_MEDIUM = '#91918d'
BOOK_CLOTH = '#cc785c'


@pytest.mark.parametrize(
    'trunc_dim, expected_layer_1_regions, expected_layer_0_regions',
    [
        (2, 2, 6),
        (3, 6, 11),
        (4, 14, 20),
        (5, 30, 42),
        (6, 60, 77),
    ],
)
def test_seqlen_2_rnn(trunc_dim: Literal[2] | Literal[3] | Literal[4] | Literal[5] | Literal[6], expected_layer_1_regions: Literal[2] | Literal[6] | Literal[14] | Literal[30] | Literal[60], expected_layer_0_regions: Literal[6] | Literal[11] | Literal[20] | Literal[42] | Literal[77]):
    """Mimics back_prop_lat_space_relu_1_post in algozoo/figures_16_10.py.

    The sequence length is 2, so there are two ReLU layers. The loop over
    'number of ReLUs - 1' therefore runs exactly once, and is written out in
    full here rather than looped.
    """
    W_hi = M_16_10_W_hi[:trunc_dim]
    W_hh = M_16_10_W_hh[:trunc_dim, :][:, :trunc_dim]
    probe_direction = M_16_10_PROBE[:trunc_dim]

    W_hh_hi = np.hstack([W_hh, W_hi])
    no_bias = np.zeros(W_hh.shape[0])

    A = probe_direction[None, :]
    regions = [ConstraintSet(
        A,
        np.full(A.shape[0], TIIINY),
        [ConstraintSet.GE] * A.shape[0]
    )]

    # ReLU layer 1: the single iteration of the loop.
    layer_1_num_regions = 0
    next_regions = []
    for region in regions:
        pre_activation_region_dict, metadata = region.reverse_relu(
            W_hh_hi, no_bias, return_metadata=True
        )
        layer_1_num_regions += metadata['num_regions']
        for pre_activation_region in pre_activation_region_dict.values():
            next_regions.append(
                pre_activation_region.reverse_add_up_proj(W_hh, W_hi, no_bias)
            )
    assert layer_1_num_regions == expected_layer_1_regions
    regions = next_regions

    # ReLU layer 0: the first layer, which needs no reverse_add_up_proj.
    layer_0_num_regions = 0
    next_regions = []
    for region in regions:
        pre_activation_region_dict, metadata = region.reverse_relu(
            W_hi, no_bias, return_metadata=True
        )
        layer_0_num_regions += metadata['num_regions']
        for pre_activation_region in pre_activation_region_dict.values():
            next_regions.append(
                pre_activation_region.reverse_up_proj(W_hi, no_bias)
            )
    assert layer_0_num_regions == expected_layer_0_regions


# ---------------------------------------------------------------------------
# What reverse_relu_with_pruning prunes on
#
# The pruned search needs a feasibility test that is monotone in the set of
# non-zeroed dims, written F: if the test fails for an orthant it must fail for
# every orthant whose F is a subset of that one. It prunes on the post-image test,
# which asks whether any post-activation point h sits in the orthant's face, with
# h >= 0 on F, h == 0 on Z, and the region's own constraints satisfied.
#
# That test is monotone because moving a dim from Z to F only relaxes h_j == 0 into
# h_j >= 0, so a witness for F is still a witness for any superset of F. It is also
# implied by the pre-image region being non-empty, so pruning on it cannot discard
# an orthant reverse_relu would have returned. Neither property mentions W, so this
# holds at both ReLU layers, including layer 0 where W_hi has rank 1.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _pipeline(trunc_dim):
    """The truncated weights and the ReLU layer 1 output regions.

    Cached because several tests below need the same regions and rebuilding them
    costs one linear program per orthant.
    """
    W_hi = M_16_10_W_hi[:trunc_dim]
    W_hh = M_16_10_W_hh[:trunc_dim, :][:, :trunc_dim]
    W_hh_hi = np.hstack([W_hh, W_hi])
    no_bias = np.zeros(W_hh.shape[0])

    A = M_16_10_PROBE[:trunc_dim][None, :]
    root = ConstraintSet(A, np.full(A.shape[0], TIIINY), [ConstraintSet.GE] * A.shape[0])

    pre_activation_region_dict = root.reverse_relu(W_hh_hi, no_bias)
    layer_1_regions = [
        pre_activation_region.reverse_add_up_proj(W_hh, W_hi, no_bias)
        for pre_activation_region in pre_activation_region_dict.values()
    ]
    return W_hi, W_hh_hi, no_bias, root, layer_1_regions


def _mask_of(zeroed_dim_idxs, trunc_dim):
    """Encode an orthant as an integer, bit i set exactly when dim i is NOT zeroed."""
    return sum(1 << i for i in range(trunc_dim) if i not in zeroed_dim_idxs)


def _post_image_feasible_masks(region, W, bias, trunc_dim):
    """Which orthants have a non-empty post image, as F masks, by brute force."""
    setup = region._reverse_relu_setup(W, bias)
    d_large, total_width, A_eq, b_eq, z_to_leq_zero_constraint = setup

    feasible = set()
    for s in range(1 << d_large):
        zeroed_dim_idxs = [i for i in range(d_large) if not (s >> i) & 1]
        _, post_image_feasible = region.reverse_relu_for_Z_tuple(
            d_large,
            zeroed_dim_idxs,
            z_to_leq_zero_constraint,
            A_eq,
            b_eq,
            total_width=total_width,
            return_post_image_feasible=True,
        )
        if post_image_feasible:
            feasible.add(s)
    return feasible


def _up_closure_violations(feasible, trunc_dim):
    """Pairs (feasible orthant, infeasible orthant with one more non-zeroed dim).

    Every such pair is a counterexample to the rule reverse_relu_with_pruning relies
    on, because it shows a feasible F whose superset is infeasible.
    """
    violations = []
    for s in sorted(feasible):
        for i in range(trunc_dim):
            if not (s >> i) & 1 and (s | (1 << i)) not in feasible:
                violations.append((s, s | (1 << i)))
    return violations


@pytest.mark.parametrize('trunc_dim', [2, 3, 4])
def test_post_image_feasibility_is_monotone_at_layer_1(trunc_dim: Literal[2] | Literal[3] | Literal[4]):
    """W_hh_hi has full row rank. The post image test is monotone here."""
    _, W_hh_hi, no_bias, root, _ = _pipeline(trunc_dim)

    feasible = _post_image_feasible_masks(root, W_hh_hi, no_bias, trunc_dim)

    # Both bounds matter. If every orthant qualified, or none did, up closure would
    # hold for trivial reasons and this test would assert nothing.
    assert 0 < len(feasible) < 2 ** trunc_dim
    assert _up_closure_violations(feasible, trunc_dim) == []


@pytest.mark.parametrize('trunc_dim', [2, 3, 4])
def test_post_image_feasibility_is_monotone_at_layer_0(trunc_dim: Literal[2] | Literal[3] | Literal[4]):
    """W_hi has rank 1, and the post image test is monotone there anyway.

    This is the case that the pre-image test gets wrong. The pre-image is pinned to
    the line spanned by W_hi, so its feasible orthants are not upward closed, but the
    post image never refers to W and stays monotone.
    """
    W_hi, _, no_bias, _, layer_1_regions = _pipeline(trunc_dim)

    assert np.linalg.matrix_rank(W_hi) < trunc_dim

    for region in layer_1_regions:
        feasible = _post_image_feasible_masks(region, W_hi, no_bias, trunc_dim)
        assert _up_closure_violations(feasible, trunc_dim) == []


@pytest.mark.parametrize('trunc_dim', [2, 3, 4, 5, 6])
def test_reverse_relu_with_pruning_matches_exhaustive_at_layer_1(trunc_dim: Literal[2] | Literal[3] | Literal[4] | Literal[5] | Literal[6]):
    """Pruning must not change the answer."""
    _, W_hh_hi, no_bias, root, _ = _pipeline(trunc_dim)

    exhaustive, exhaustive_metadata = root.reverse_relu(
        W_hh_hi, no_bias, return_metadata=True
    )
    pruned, pruned_metadata = root.reverse_relu_with_pruning(
        W_hh_hi, no_bias, return_metadata=True
    )

    assert set(exhaustive) == set(pruned)
    for z_tuple in exhaustive:
        assert np.array_equal(exhaustive[z_tuple].A, pruned[z_tuple].A)
        assert np.array_equal(exhaustive[z_tuple].b, pruned[z_tuple].b)

    # One call per orthant with a non-empty post image, plus one per maximal orthant
    # without one, and nothing else.
    assert pruned_metadata['num_verifier_calls'] == (
        pruned_metadata['num_post_image_feasible']
        + pruned_metadata['num_maximal_infeasible']
    )
    assert pruned_metadata['num_verifier_calls'] <= exhaustive_metadata['num_total_regimes']


@pytest.mark.parametrize('trunc_dim', [2, 3, 4, 5, 6])
def test_reverse_relu_with_pruning_matches_exhaustive_at_layer_0(trunc_dim: Literal[2] | Literal[3] | Literal[4] | Literal[5] | Literal[6]):
    """The rank-1 layer, where pruning on the pre-image would have lost regions."""
    W_hi, _, no_bias, _, layer_1_regions = _pipeline(trunc_dim)

    for region in layer_1_regions:
        exhaustive = region.reverse_relu(W_hi, no_bias)
        pruned, pruned_metadata = region.reverse_relu_with_pruning(
            W_hi, no_bias, return_metadata=True
        )

        assert set(exhaustive) == set(pruned)
        for z_tuple in exhaustive:
            assert np.array_equal(exhaustive[z_tuple].A, pruned[z_tuple].A)
            assert np.array_equal(exhaustive[z_tuple].b, pruned[z_tuple].b)
        assert pruned_metadata['num_verifier_calls'] == (
            pruned_metadata['num_post_image_feasible']
            + pruned_metadata['num_maximal_infeasible']
        )


# ---------------------------------------------------------------------------
# Spreading a layer's regions over workers
#
# reverse_relu_layer splits on regions rather than on orthants, so a region's
# pruned search never straddles two workers and its greedy `max_infeas` is never
# shared. That is what makes the parallel answer identical to the serial one
# rather than merely equivalent, and it is what these check: same orthants, same
# constraints, and the same number of verifier calls, which would move if a worker
# had lost prunes another worker found.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('trunc_dim', [4, 5])
def test_reverse_relu_layer_serial_matches_the_method(trunc_dim: Literal[4] | Literal[5]):
    """With no executor it is the list comprehension it replaces."""
    W_hi, _, no_bias, _, layer_1_regions = _pipeline(trunc_dim)

    expected = [
        region.reverse_relu_with_pruning(W_hi, no_bias)
        for region in layer_1_regions
    ]
    actual = reverse_relu_layer(layer_1_regions, W_hi, no_bias)

    assert len(actual) == len(expected)
    for i, (expected_dict, actual_dict) in enumerate(zip(expected, actual)):
        _assert_same_regions(expected_dict, actual_dict, f'region {i}')


@pytest.mark.parametrize('pruning', [True, False])
def test_reverse_relu_layer_parallel_matches_serial(pruning: bool):
    """Two processes return what one process did, region for region and in order."""
    trunc_dim = 5
    W_hi, _, no_bias, _, layer_1_regions = _pipeline(trunc_dim)

    serial, serial_metadata = reverse_relu_layer(
        layer_1_regions, W_hi, no_bias, pruning=pruning, return_metadata=True
    )
    parallel, parallel_metadata = reverse_relu_layer(
        layer_1_regions, W_hi, no_bias, pruning=pruning, executor=2,
        return_metadata=True,
    )

    assert serial_metadata['num_workers'] == 1
    assert parallel_metadata['num_workers'] == 2
    for i, (serial_dict, parallel_dict) in enumerate(zip(serial, parallel)):
        _assert_same_regions(serial_dict, parallel_dict, f'region {i}')

    # The counters, unlike the seconds, are a property of the search and not of
    # the machine. Pruning that had been lost by splitting a region's search
    # across workers would show up here as extra verifier calls.
    for counter in ('num_regions', 'num_total_regimes'):
        assert parallel_metadata[counter] == serial_metadata[counter]
    if pruning:
        assert parallel_metadata['num_verifier_calls'] == \
            serial_metadata['num_verifier_calls']


def test_reverse_relu_layer_metadata_aggregates():
    """The aggregate is the sum of the parts, and says how many parts there were."""
    trunc_dim = 4
    W_hi, _, no_bias, _, layer_1_regions = _pipeline(trunc_dim)

    parts = [
        region.reverse_relu_with_pruning(W_hi, no_bias, return_metadata=True)[1]
        for region in layer_1_regions
    ]
    _, aggregate = reverse_relu_layer(
        layer_1_regions, W_hi, no_bias, return_metadata=True
    )

    assert aggregate['num_regions_in'] == len(layer_1_regions)
    for counter in ('num_regions', 'num_total_regimes', 'num_verifier_calls'):
        assert aggregate[counter] == sum(part[counter] for part in parts)
    assert aggregate['wall_clock_seconds'] >= 0.


def test_flatten_layer_applies_the_substitution():
    """flatten_layer is flatten_region_dicts over a whole layer, plus substitution."""
    trunc_dim = 4
    W_hi, _, no_bias, _, layer_1_regions = _pipeline(trunc_dim)

    region_dicts = reverse_relu_layer(layer_1_regions, W_hi, no_bias)
    expected = [
        region
        for region_dict in region_dicts
        for region in region_dict.values()
    ]

    assert flatten_layer(region_dicts) == expected

    substituted = flatten_layer(
        region_dicts, lambda region: region.reverse_up_proj(W_hi, no_bias)
    )
    assert len(substituted) == len(expected)
    for region, pre_activation_region in zip(substituted, expected):
        assert np.array_equal(
            region.A, pre_activation_region.reverse_up_proj(W_hi, no_bias).A
        )


def test_executor_close_shuts_the_workers_down():
    """close() must actually reap the processes, not just drop joblib's handle.

    joblib keeps its loky workers warm between calls on purpose, so releasing
    Parallel leaves them running. Workers still up when the interpreter starts
    tearing down are what provokes the ChildProcessError traceback CPython 3.12
    prints from ResourceTracker.__del__, which is the whole reason close() exists.
    """
    from joblib.externals.loky import get_reusable_executor

    from lat_space_lens import JoblibExecutor, shutdown_workers

    def num_alive():
        processes = getattr(get_reusable_executor(reuse=True), '_processes', None)
        return sum(process.is_alive() for process in (processes or {}).values())

    with JoblibExecutor(n_jobs=2) as executor:
        executor.map(abs, [-1, -2, -3])
        assert num_alive() > 0
    assert num_alive() == 0

    # Idempotent, and not final: mapping again brings a pool back up.
    executor.close()
    assert executor.map(abs, [-4]) == [4]
    assert num_alive() > 0
    shutdown_workers()
    assert num_alive() == 0
    shutdown_workers()


def test_closing_stops_the_resource_tracker():
    """close() must leave multiprocessing's resource tracker stopped.

    CPython 3.12's ResourceTracker.__del__ waits on the tracker child and prints an
    uncatchable ChildProcessError traceback at interpreter shutdown when that child
    is already gone. The finalizer returns early when _fd and _pid are None, so
    stopping the tracker while the workers are down -- where an error is an ordinary
    catchable one -- is what keeps it quiet. See parallel.stop_resource_tracker.
    """
    from multiprocessing import resource_tracker

    from lat_space_lens import JoblibExecutor, stop_resource_tracker

    tracker = resource_tracker._resource_tracker

    with JoblibExecutor(n_jobs=2) as executor:
        assert executor.map(abs, [-1, -2, -3]) == [1, 2, 3]
        # Spawning the pool is what starts the tracker, so this is the state the
        # finalizer would later trip over.
        assert tracker._fd is not None and tracker._pid is not None
    assert tracker._fd is None and tracker._pid is None

    # Which is exactly the state the finalizer returns early on, so running it
    # by hand raises nothing.
    type(tracker).__del__(tracker)

    # Nothing is permanently gone: more parallel work starts a tracker again.
    assert executor.map(abs, [-4]) == [4]
    assert tracker._pid is not None
    executor.close()
    assert tracker._pid is None

    # And with none running there is nothing to stop, said plainly rather than raised.
    assert stop_resource_tracker() is False


def test_resolve_executor():
    """None and 1 mean the serial loop itself, not a one-worker pool."""
    from lat_space_lens import JoblibExecutor, SerialExecutor, resolve_executor

    assert isinstance(resolve_executor(None), SerialExecutor)
    assert isinstance(resolve_executor(1), SerialExecutor)

    executor = resolve_executor(3)
    assert isinstance(executor, JoblibExecutor)
    assert executor.num_workers == 3

    assert resolve_executor(executor) is executor
    with pytest.raises(TypeError):
        resolve_executor('4')


# ---------------------------------------------------------------------------
# Timing, run from the terminal rather than under pytest
# ---------------------------------------------------------------------------


def _print_timing_row(
        trunc_dim, layer, exhaustive_seconds, pruned_seconds, regimes, calls, regions):
    speedup = exhaustive_seconds / pruned_seconds if pruned_seconds else float('nan')
    print(f'{trunc_dim:>5} {layer:>5} {exhaustive_seconds:>13.4f} {pruned_seconds:>10.4f} '
          f'{speedup:>7.2f}x {regimes:>8} {calls:>7} {regions:>8}')


def short_timing_report(trunc_dims=(9, 10), skip_exhaustive=False, executor=None):
    """Print exhaustive against pruned wall clock, taken from return_metadata.

    This asserts nothing and pytest does not collect it, because its numbers depend
    on the machine and on what else is running. The times are the summed
    'total_time_taken' the two methods report, which covers the orthant loop only
    and excludes setup. Layer 0 is summed over every region that layer 1 produced,
    and its regions are spread over `executor` -- see reverse_relu_layer. Layer 1
    starts from a single region, so there is nothing there to spread.

    'regions' counts the calls that produced a region rather than None. Pruned and
    exhaustive return the same regions, so it is read off the pruned run, which is
    the one that always happens.

    @param executor:    None or 1 for a serial loop, an int for that many joblib
                        workers, or a lat_space_lens Executor
    """
    print(f'{"trunc":>5} {"layer":>5} {"exhaustive s":>13} {"pruned s":>10} '
          f'{"speedup":>8} {"regimes":>8} {"calls":>7} {"regions":>8}')

    # Resolved once and closed at the end, so the workers are started once for the
    # whole report rather than per layer, and are gone before the interpreter
    # exits. See Executor.close.
    executor = resolve_executor(executor)
    with executor:
        _short_timing_rows(trunc_dims, skip_exhaustive, executor)


def _short_timing_rows(trunc_dims, skip_exhaustive, executor):
    for trunc_dim in trunc_dims:
        W_hi, W_hh_hi, no_bias, root, layer_1_regions = _pipeline(trunc_dim)

        if not skip_exhaustive:
            _, exhaustive = root.reverse_relu(W_hh_hi, no_bias, return_metadata=True)
        _, pruned = root.reverse_relu_with_pruning(W_hh_hi, no_bias, return_metadata=True)
        _print_timing_row(
            trunc_dim,
            1,
            0.0 if skip_exhaustive else exhaustive['total_time_taken'],
            pruned['total_time_taken'],
            pruned['num_total_regimes'],
            pruned['num_verifier_calls'],
            pruned['num_regions'],
        )

        exhaustive_seconds = 0.0
        if not skip_exhaustive:
            _, exhaustive = reverse_relu_layer(
                layer_1_regions, W_hi, no_bias, pruning=False, executor=executor,
                return_metadata=True,
            )
            exhaustive_seconds = exhaustive['total_cpu_seconds']
        _, pruned = reverse_relu_layer(
            layer_1_regions, W_hi, no_bias, pruning=True, executor=executor,
            return_metadata=True,
        )
        _print_timing_row(
            trunc_dim,
            0,
            exhaustive_seconds,
            pruned['total_cpu_seconds'],
            pruned['num_total_regimes'],
            pruned['num_verifier_calls'],
            pruned['num_regions'],
        )


# ---------------------------------------------------------------------------
# The same timing, over contiguous submatrices sampled from the whole zoo
# ---------------------------------------------------------------------------

_W_HH_NAME_RE = re.compile(
    r'^(?P<prefix>(?:M|ZOO)_(?P<dim>\d+)_(?P<seq_len>\d+))_W_hh$'
)


@lru_cache(maxsize=None)
def _available_models():
    """(seq_len, dim) -> constant name prefix, for every model in constants.py.

    Read off the constant names rather than listed here, so that regenerating
    constants.py over a different grid needs no change in this file.

    A model counts only if it carries probes under the _PROBE_RELU_{L}
    convention. That excludes M_16_10, whose single probe is named M_16_10_PROBE
    because it is the example checkpoint rather than a zoo one, and which would
    otherwise collide with ZOO_16_10 on the key (10, 16).
    """
    models = {}
    for name in dir(constants):
        match = _W_HH_NAME_RE.match(name)
        if match is None:
            continue
        prefix = match.group('prefix')
        if not hasattr(constants, f'{prefix}_PROBE_RELU_0'):
            continue
        key = (int(match.group('seq_len')), int(match.group('dim')))
        assert key not in models, \
            f'two models share the key {key}: {models[key]} and {prefix}'
        models[key] = prefix
    return models


def _submatrix_triplets(seq_len, dim):
    """The (model_seq_len, model_dim, start_dim_idx) triplets this search can use.

    A model qualifies when it has at least `seq_len` ReLU layers, at least `dim`
    hidden dims, and the relu_{seq_len - 1}_post probe that the root constraint
    is built from. Each qualifying model contributes one triplet per contiguous
    dim-wide window of its dims, so the triplet names a unique submatrix.
    """
    triplets = []
    for (model_seq_len, model_dim), prefix in _available_models().items():
        if model_seq_len < seq_len or model_dim < dim:
            continue
        if not hasattr(constants, f'{prefix}_PROBE_RELU_{seq_len - 1}'):
            continue
        for start_dim_idx in range(0, model_dim - dim + 1):
            triplets.append((model_seq_len, model_dim, start_dim_idx))
    return sorted(triplets)


def _truncated_weights(triplet, seq_len, dim):
    """One triplet's contiguous submatrices, and the root constraint over them.

    The window is taken out of W_hh, W_hi and the probe alike, so the truncated
    weights are those of a well-defined smaller recurrence. It is not the model
    that would be trained at this size, and the probe is the full model's probe
    restricted to the window rather than one fitted on it: as in the tests above,
    these numbers are about the size of the linear-programming search, not about
    what the truncated network computes.
    """
    model_seq_len, model_dim, start_dim_idx = triplet
    prefix = _available_models()[(model_seq_len, model_dim)]
    stop = start_dim_idx + dim

    W_hi = getattr(constants, f'{prefix}_W_hi')[start_dim_idx:stop]
    W_hh = getattr(
        constants, f'{prefix}_W_hh'
    )[start_dim_idx:stop, :][:, start_dim_idx:stop]
    probe_direction = getattr(
        constants, f'{prefix}_PROBE_RELU_{seq_len - 1}'
    )[start_dim_idx:stop]

    W_hh_hi = np.hstack([W_hh, W_hi])
    no_bias = np.zeros(W_hh.shape[0])

    A = probe_direction[None, :]
    root = ConstraintSet(
        A,
        np.full(A.shape[0], TIIINY),
        [ConstraintSet.GE] * A.shape[0]
    )
    return W_hi, W_hh, W_hh_hi, no_bias, root


def _assert_same_regions(exhaustive, pruned, context):
    """Enforce that pruning returned exactly what the exhaustive search did.

    Same orthants, and for each one the same region: pruning is only allowed to
    skip work, never to change the answer. Both paths build their regions in
    reverse_relu_for_Z_tuple from the same arguments, so every field should agree
    bitwise, not merely to a tolerance.

    This is what test_reverse_relu_with_pruning_matches_exhaustive_at_layer_1 and
    _at_layer_0 assert, checked here on the sampled submatrices instead of on the
    one model those tests use.
    """
    assert set(exhaustive) == set(pruned), (
        f'{context}: exhaustive and pruned disagree on which orthants have a '
        f'region, only exhaustive has {sorted(set(exhaustive) - set(pruned))}, '
        f'only pruned has {sorted(set(pruned) - set(exhaustive))}'
    )
    for z_tuple in exhaustive:
        for field in ('A', 'b', 'inequalities', 'block_sizes'):
            exhaustive_value = getattr(exhaustive[z_tuple], field)
            pruned_value = getattr(pruned[z_tuple], field)
            assert np.array_equal(exhaustive_value, pruned_value), (
                f'{context}: orthant {z_tuple} differs in {field}, '
                f'exhaustive {exhaustive_value}, pruned {pruned_value}'
            )


def _reverse_one_layer(
        regions,
        W,
        no_bias,
        substitute,
        skip_exhaustive,
        test_equality=False,
        context='',
        executor=None,
    ):
    """Reverse one ReLU layer over every region, exhaustive against pruned.

    `substitute` rewrites one pre-activation region over the variable the next
    layer back starts from: reverse_add_up_proj at every layer that has a
    previous hidden state to reverse into, reverse_up_proj at layer 0.

    The two methods return the same regions, so the ones that propagate are read
    off the pruned run, which is the one that always happens. Under
    `test_equality` that sameness stops being an assumption and is checked on
    every region, which needs the exhaustive run and so rules out
    `skip_exhaustive`.

    Both searches go through reverse_relu_layer, which spreads the regions over
    `executor`. Each region's pruned search still runs whole inside one worker,
    so the regions that come back do not depend on how many workers there were;
    the timings do. See lat_space_lens/parallel.py.

    @param executor:    None or 1 for a serial loop, an int for that many joblib
                        workers, or an Executor
    """
    assert not (test_equality and skip_exhaustive), \
        'test_equality needs the exhaustive regions to compare against'

    exhaustive_seconds = pruned_seconds = 0.
    exhaustive_wall_seconds = pruned_wall_seconds = 0.
    exhaustive_region_dicts = None
    if not skip_exhaustive:
        exhaustive_region_dicts, exhaustive = reverse_relu_layer(
            regions, W, no_bias, pruning=False, executor=executor,
            return_metadata=True,
        )
        exhaustive_seconds = exhaustive['total_cpu_seconds']
        exhaustive_wall_seconds = exhaustive['wall_clock_seconds']

    pre_activation_region_dicts, pruned = reverse_relu_layer(
        regions, W, no_bias, pruning=True, executor=executor,
        return_metadata=True,
    )
    pruned_seconds = pruned['total_cpu_seconds']
    pruned_wall_seconds = pruned['wall_clock_seconds']

    if test_equality:
        for exhaustive_region_dict, pre_activation_region_dict in zip(
                exhaustive_region_dicts, pre_activation_region_dicts):
            _assert_same_regions(
                exhaustive_region_dict, pre_activation_region_dict, context
            )

    next_regions = flatten_layer(pre_activation_region_dicts, substitute)

    row = {
        'regions_in': len(regions),
        'exhaustive_seconds': exhaustive_seconds,
        'pruned_seconds': pruned_seconds,
        'exhaustive_wall_seconds': exhaustive_wall_seconds,
        'pruned_wall_seconds': pruned_wall_seconds,
        'num_workers': pruned['num_workers'],
        'num_total_regimes': pruned['num_total_regimes'],
        'num_verifier_calls': pruned['num_verifier_calls'],
        'num_regions': pruned['num_regions'],
    }
    return next_regions, row


def _print_full_timing_row(triplet, layer, row):
    """One line of the report.

    'pruned s' is the summed cost of the searches and 'wall s' is what the layer
    actually took, so with one worker they are about equal and with several the
    gap between them is what the executor bought. The speedup column stays the
    ratio of the two searches' cost, which is what pruning bought, so that a run
    on many workers is still comparable with a serial one.
    """
    model_seq_len, model_dim, start_dim_idx = triplet
    pruned_seconds = row['pruned_seconds']
    speedup = (
        row['exhaustive_seconds'] / pruned_seconds
        if pruned_seconds else float('nan')
    )
    print(f"{model_seq_len:>4} {model_dim:>4} {start_dim_idx:>6} {layer:>6} "
          f"{row['regions_in']:>8} {row['exhaustive_seconds']:>13.4f} "
          f"{pruned_seconds:>10.4f} {speedup:>7.2f}x "
          f"{row['pruned_wall_seconds']:>8.4f} {row['num_workers']:>7} "
          f"{row['num_total_regimes']:>9} {row['num_verifier_calls']:>8} "
          f"{row['num_regions']:>8}")


def full_timing_report(
        seq_len,
        dim,
        n=3,
        seed=42,
        skip_exhaustive=False,
        test_equality: bool = False,
        executor=None,
        save_to=None,
    ):
    """short_timing_report over submatrices sampled from every model in the zoo.

    short_timing_report walks one model at a handful of truncation widths. This
    walks `n` randomly chosen contiguous dim-wide windows drawn from every model
    with room for the search, so the timings are not all read off the same
    weights.

    The eligible windows are the (model_seq_len, model_dim, start_dim_idx)
    triplets of _submatrix_triplets: every model with at least `seq_len` ReLU
    layers and at least `dim` hidden dims contributes one triplet per start
    index. `n` of them are sampled without replacement, seeded by `seed`.

    Each sampled window is reversed through all `seq_len` ReLU layers, starting
    from the relu_{seq_len - 1}_post probe half-space. Layers seq_len - 1 down to
    1 substitute with reverse_add_up_proj, which injects that layer's input
    coordinate as a frozen block; layer 0 has no previous hidden state and
    substitutes with reverse_up_proj. Every layer therefore reverses a ReLU on
    `dim` dims, so each region costs up to 2 ** dim orthants, and the region
    count multiplies from one layer to the next. Both grow fast: `dim` in the
    teens with more than a couple of layers will not finish.

    The times are the 'total_time_taken' the two methods report, which covers the
    orthant loop only and excludes setup, summed over every region entering the
    layer. This asserts nothing and pytest does not collect it, because its
    numbers depend on the machine and on what else is running.

    @param seq_len:         number of ReLU layers to reverse. At most 4, because
                            the root is the relu_{seq_len - 1}_post probe and
                            constants.py carries probes for layers 0 to 3
    @param dim:             width of the contiguous window to take
    @param n:               how many windows to sample
    @param seed:            seeds the sample
    @param skip_exhaustive: only run the pruned search, leaving the exhaustive
                            column at 0 and the speedup meaningless
    @param test_equality:   check at every layer that the pruned search returned
                            exactly the regions the exhaustive one did, the same
                            orthants with bitwise identical constraints. Raises
                            on the first disagreement. Needs the exhaustive
                            regions, so it cannot be combined with
                            `skip_exhaustive`
    @param executor:        how to spread each layer's regions over workers:
                            None or 1 for a serial loop, an int for that many
                            joblib workers, or a lat_space_lens Executor. The
                            regions that come back do not depend on this, only
                            the wall clock does
    @param save_to:         where to write the JSON record, which holds the
                            arguments, the sampled triplets and every row.
                            Defaults to a name built from the arguments, in the
                            current working directory. None of it is printed
                            that is not also saved
    @return:                the path written to
    """
    if seq_len < 1:
        raise ValueError(f'seq_len must be at least 1, got {seq_len}')
    if dim < 1:
        raise ValueError(f'dim must be at least 1, got {dim}')
    if test_equality and skip_exhaustive:
        raise ValueError(
            'test_equality compares the pruned regions against the exhaustive '
            'ones, so it cannot be used with skip_exhaustive'
        )

    triplets = _submatrix_triplets(seq_len, dim)
    if not triplets:
        raise ValueError(
            f'no model in constants.py has at least {seq_len} ReLU layers, at '
            f'least {dim} hidden dims and a relu_{seq_len - 1}_post probe'
        )
    if n > len(triplets):
        raise ValueError(
            f'asked for {n} submatrices but only {len(triplets)} exist for '
            f'seq_len={seq_len}, dim={dim}'
        )
    sampled = sorted(random.Random(seed).sample(triplets, n))

    if save_to is None:
        save_to = f'full_timing_report_seq{seq_len}_dim{dim}_n{n}_seed{seed}.json'
    save_to = Path(save_to)

    print(f'seq_len={seq_len} dim={dim} n={n} seed={seed} '
          f'skip_exhaustive={skip_exhaustive} test_equality={test_equality} '
          f'executor={executor}')
    print(f'sampled {n} of {len(triplets)} eligible submatrices: {sampled}')
    print(f'{"mseq":>4} {"mdim":>4} {"start":>6} {"layer":>6} {"regions in":>8} '
          f'{"exhaustive s":>13} {"pruned s":>10} {"speedup":>8} '
          f'{"wall s":>8} {"workers":>7} '
          f'{"regimes":>9} {"calls":>8} {"regions":>8}')

    # Resolved once and closed at the end, so the workers are started once for the
    # whole report rather than per layer of every triplet, and are gone before the
    # interpreter exits. See Executor.close.
    executor = resolve_executor(executor)
    with executor:
        rows = _full_timing_rows(
            sampled, seq_len, dim, skip_exhaustive, test_equality, executor
        )

    record = {
        'seq_len': seq_len,
        'dim': dim,
        'n': n,
        'seed': seed,
        'skip_exhaustive': skip_exhaustive,
        'test_equality': test_equality,
        'executor': repr(executor),
        'num_eligible_triplets': len(triplets),
        'sampled_triplets': [list(triplet) for triplet in sampled],
        'rows': rows,
    }
    save_to.parent.mkdir(parents=True, exist_ok=True)
    with open(save_to, 'w') as f:
        json.dump(record, f, indent=2)
    print(f'\nwrote {save_to}')
    return save_to


def _full_timing_rows(sampled, seq_len, dim, skip_exhaustive, test_equality, executor):
    """The body of full_timing_report's loop, one row per triplet per layer."""
    rows = []
    for triplet in sampled:
        W_hi, W_hh, W_hh_hi, no_bias, root = _truncated_weights(
            triplet, seq_len, dim
        )
        regions = [root]

        # Layers seq_len - 1 down to 1. Each one reverses the ReLU over the joint
        # variable [h_{t-1}; x_t] and then freezes x_t into the injected suffix.
        for layer in range(seq_len - 1, 0, -1):
            regions, row = _reverse_one_layer(
                regions,
                W_hh_hi,
                no_bias,
                lambda pre_activation_region: (
                    pre_activation_region.reverse_add_up_proj(W_hh, W_hi, no_bias)
                ),
                skip_exhaustive,
                test_equality=test_equality,
                context=f'{triplet} layer {layer}',
                executor=executor,
            )
            _print_full_timing_row(triplet, layer, row)
            rows.append({
                'model_seq_len': triplet[0],
                'model_dim': triplet[1],
                'start_dim_idx': triplet[2],
                'layer': layer,
                **row,
            })

        # Layer 0 has no previous hidden state, so there is nothing to inject and
        # the substitution is a plain reverse_up_proj.
        regions, row = _reverse_one_layer(
            regions,
            W_hi,
            no_bias,
            lambda pre_activation_region: (
                pre_activation_region.reverse_up_proj(W_hi, no_bias)
            ),
            skip_exhaustive,
            test_equality=test_equality,
            context=f'{triplet} layer 0',
            executor=executor,
        )
        _print_full_timing_row(triplet, 0, row)
        rows.append({
            'model_seq_len': triplet[0],
            'model_dim': triplet[1],
            'start_dim_idx': triplet[2],
            'layer': 0,
            **row,
        })
    return rows


def _mean_and_ci_half_width(values, confidence):
    """Sample mean, and the half width of its two-sided confidence interval.

    The t distribution rather than the normal one, because these reports carry
    very few samples per layer: at the default n of 3, using 1.96 standard errors
    would understate the interval by a factor of about 2.2.

    One sample has no spread to estimate from, so its half width is 0. That is a
    missing error bar, not a claim that the mean is exact.
    """
    from scipy.stats import t as t_distribution

    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(values.mean())
    if n < 2:
        return mean, 0.
    standard_error = float(values.std(ddof=1) / np.sqrt(n))
    half_width = float(
        t_distribution.ppf(0.5 + confidence / 2, n - 1) * standard_error
    )
    return mean, half_width


def analyze_full_timing_report(report_path, save_to=None, confidence=0.95):
    """Plot a full_timing_report's two searches against how far back they have gone.

    The horizontal axis is the number of layer propagations. The ReLU layers are
    reversed from the probe backwards, so layer seq_len - 1 is the first one
    reached and sits at 0, and layer 0 is the last and sits at seq_len - 1. For a
    4 layer report the layers 3, 2, 1, 0 therefore land on the ticks 0, 1, 2, 3.

    Each tick carries one point per series: the mean, over the sampled
    submatrices, of the seconds that layer took, with error bars at the given
    confidence level. The two series are the exhaustive search and the pruned one,
    so the gap between them is what pruning bought at that depth.

    A report that holds no exhaustive timings, because it was written with
    skip_exhaustive or because its rows do not carry the field, is plotted from
    the pruned series alone rather than refused. Only a report with nothing to
    plot at all is an error.

    @param report_path: the JSON written by full_timing_report
    @param save_to:     where to write the figure. Defaults to the report's own
                        path with a .png suffix
    @param confidence:  the level for the error bars, 0.95 by default
    @return:            the path written to
    """
    # Imported here rather than at module scope so that collecting this file
    # under pytest does not pay for matplotlib.
    from matplotlib import pyplot as plt

    report_path = Path(report_path)
    with open(report_path) as f:
        record = json.load(f)

    if not record['rows']:
        raise ValueError(f'{report_path} has no rows')

    seq_len = record['seq_len']
    rows_by_position = {}
    for row in record['rows']:
        # Reversal starts at the probe, so the highest layer is the first reached.
        position = seq_len - 1 - row['layer']
        rows_by_position.setdefault(position, []).append(row)
    positions = sorted(rows_by_position)

    # A run with skip_exhaustive never measured the exhaustive search and wrote
    # 0.0 in its place, so its rows carry the field but not the timing. Drop the
    # series rather than draw a line pinned to zero and label it a measurement.
    series = []
    if not record.get('skip_exhaustive', False):
        series.append(('exhaustive', 'exhaustive_seconds', BOOK_CLOTH))
    series.append(('pruned', 'pruned_seconds', CLOUD_MEDIUM))

    plotted = []
    for label, key, color in series:
        means = []
        half_widths = []
        for position in positions:
            # A row from another producer need not carry every field, so take
            # the mean over the rows that do have it.
            values = [
                row[key] for row in rows_by_position[position] if key in row
            ]
            if not values:
                # NaN leaves a gap in the line here, rather than a point at zero
                # that would read as a layer that took no time.
                means.append(np.nan)
                half_widths.append(np.nan)
                continue
            mean, half_width = _mean_and_ci_half_width(values, confidence)
            means.append(mean)
            half_widths.append(half_width)
        if not all(np.isnan(mean) for mean in means):
            plotted.append((label, color, means, half_widths))

    if not plotted:
        raise ValueError(f'{report_path} carries no timings to plot')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor(PAPER)
    # Series carry near-identical times at the shallow end, so their error bars
    # would be drawn on top of each other. Nudge them apart by a fraction of a
    # tick; the ticks themselves stay on the integers. A lone series is not
    # nudged, because there is nothing for it to collide with.
    dodges = np.linspace(-0.04, 0.04, len(plotted)) if len(plotted) > 1 else [0.]
    for (label, color, means, half_widths), dodge in zip(plotted, dodges):
        ax.errorbar(
            [position + dodge for position in positions],
            means,
            yerr=half_widths,
            label=label,
            color=color,
            marker='o',
            markersize=5,
            capsize=4,
            linewidth=1.5,
            elinewidth=1.,
        )

    percent = round(confidence * 100)
    ax.set_xlabel('num_layer_propagations')
    ax.set_ylabel('seconds')
    ax.set_title(
        f"seq_len={seq_len} dim={record['dim']} n={record['n']} "
        f"seed={record['seed']}\nmean orthant loop seconds, {percent}% CI",
        fontsize=9
    )
    ax.set_xticks(positions)
    # Deliberately not clamped to 0. When the spread is wide enough that the
    # interval reaches below zero, clipping it there would hide half the error
    # bar and read as a tighter estimate than the samples support.
    ax.axhline(0, color=CLOUD_LIGHT, linewidth=0.5)
    ax.legend()
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    if save_to is None:
        save_to = report_path.with_suffix('.png')
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=220, facecolor=fig.get_facecolor())

    n_per_position = [len(rows_by_position[position]) for position in positions]
    print(f'{report_path}: seq_len={seq_len} dim={record["dim"]} '
          f'n={record["n"]} seed={record["seed"]}')
    if not any(label == 'exhaustive' for label, *_ in plotted):
        print('  no exhaustive timings in this report, so the pruned series is '
              'plotted alone')
    if min(n_per_position) < 2:
        print(f'  some positions have one sample, so their {percent}% CI half '
              f'width is 0 rather than estimated')
    header = f'{"props":>5} {"layer":>5} {"samples":>7}'
    for label, *_ in plotted:
        header += f' {label + " s":>13} {"+-":>9}'
    print(header)
    for i, position in enumerate(positions):
        line = (f'{position:>5} {seq_len - 1 - position:>5} '
                f'{n_per_position[i]:>7}')
        for _, _, means, half_widths in plotted:
            line += f' {means[i]:>13.4f} {half_widths[i]:>9.4f}'
        print(line)
    print(f'\nwrote {save_to}')

    plt.show()
    return save_to


if __name__ == '__main__':
    # short_timing_report()
    # full_timing_report(
    #     seq_len=4,
    #     dim=6,
    #     n=5,
    #     seed=42,
    #     skip_exhaustive=False,
    #     test_equality=False,
    #     # None or 1 for a serial loop, an int for that many worker processes.
    #     # Only the first layer of a report is one region wide, so anything past
    #     # it has plenty to spread; that first layer pays the pool startup for
    #     # nothing, which is why its wall clock can exceed its cpu seconds.
    #     executor=-1,
    # )
    analyze_full_timing_report('full_timing_report_seq4_dim6_n5_seed42.json')