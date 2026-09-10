"""Test cases for the reverse-ReLU activation pattern search.

The weights below are hard-coded so that this file does not depend on the
alg_zoo library. They were extracted with:

    from alg_zoo import example_2nd_argmax
    model = example_2nd_argmax()
    W_hh = as_numpy(model.rnn.weight_hh_l0)
    W_hi = as_numpy(model.rnn.weight_ih_l0)

and, following lines 500-517 of algozoo/figures_16_10.py with seed=42, from the
'relu_1_post' linear probe fitted against the labeling inputs[1] > inputs[0] and
inputs[1] > 0.

All three arrays are stored untruncated. Each test truncates them itself.
"""
from functools import lru_cache

import numpy as np
import pytest

from lat_space_lens import ConstraintSet

# Shape (16, 16), from model.rnn.weight_hh_l0.
M_16_10_W_hh = np.array([
    [0.16085967421531677, 0.7108793258666992, -0.17373062670230865, 0.021222922950983047, 0.5948790907859802, -0.9134215712547302, 1.1327556371688843, -0.7758187651634216, -0.7929948568344116, -1.4237356185913086, 0.3074500858783722, 0.15022523701190948, 0.224845290184021, 0.07666322588920593, 0.4203847348690033, 0.00833763275295496],
    [-0.10828900337219238, 0.4093594551086426, 0.5037025213241577, 0.011614464223384857, 0.8713551759719849, 0.11774849891662598, 0.9953998327255249, -1.0476715564727783, -0.1203792467713356, -0.6862773299217224, 0.023890815675258636, 0.00844612531363964, 0.04300367832183838, 0.029531795531511307, 0.07914742827415466, 0.02005760744214058],
    [-0.083544060587883, -0.10195071250200272, 0.9688376188278198, 0.03908676654100418, 1.7307186126708984, -0.010148447938263416, -0.10253623872995377, 0.022688522934913635, 0.15453435480594635, -0.22885268926620483, 0.015223793685436249, 0.01761922799050808, 0.055331744253635406, 0.0404987595975399, 0.02686350978910923, -0.04666026681661606],
    [-1.2293846607208252, 1.1952526569366455, 0.0016773446695879102, -0.35855749249458313, 0.34419357776641846, -0.5175389647483826, 0.48519280552864075, -0.3992026150226593, -1.0741428136825562, 0.2081184983253479, 0.21385245025157928, -0.19370673596858978, 0.4988366663455963, -0.2585681080818176, 0.5884541273117065, 0.8104538917541504],
    [0.035378072410821915, 0.035622864961624146, -0.492309033870697, -0.0033940111752599478, -0.9856111407279968, 0.0016950786812230945, 0.09330034255981445, -0.029368871822953224, -0.08155543357133865, 0.08339191973209381, -0.006543969735503197, -0.012211793102324009, -0.059153247624635696, -0.026931047439575195, -0.032630957663059235, -0.010949648916721344],
    [0.2244897484779358, -0.0005086797173134983, -0.03115234524011612, -0.20049084722995758, 1.18705153465271, -0.42111334204673767, 0.015222044661641121, -0.5933382511138916, -0.8497561812400818, 0.2973608076572418, 0.12241365760564804, 0.32778841257095337, 0.2613863945007324, -0.009049180895090103, 0.20479953289031982, 0.07378725707530975],
    [-0.09288160502910614, -0.13510721921920776, 0.5349642634391785, 0.027262309566140175, -0.04709412902593613, 0.03827216103672981, 0.3605724275112152, -0.12607648968696594, -0.08202435821294785, -0.6400814652442932, 0.028036346659064293, 0.018916036933660507, 0.03955690935254097, 0.01606828346848488, 0.08612384647130966, -0.007887518964707851],
    [-0.021420316770672798, -0.10036176443099976, 0.673674464225769, -0.008009334094822407, 1.3166165351867676, 0.002307594520971179, -0.18246307969093323, 0.07815435528755188, 0.1666875183582306, 0.002781821181997657, -0.0038847674150019884, 0.029899975284934044, 0.0695122629404068, 0.02172163501381874, -1.44146633829223e-05, -0.012527556158602238],
    [-0.004610032774507999, 0.6339022517204285, 0.6253562569618225, 0.00046008595381863415, -0.04149255156517029, -0.01696033589541912, -0.00028777666739188135, -1.0941182374954224, 0.6153913140296936, -0.06835651397705078, 0.038094412535429, 0.05488774552941322, 0.04157969355583191, 0.0006598306936211884, 0.03189060091972351, -0.013365192338824272],
    [-0.03175472095608711, -0.3573145270347595, 0.004240703769028187, -0.04386823996901512, 0.14580999314785004, 0.042859479784965515, -0.08918856084346771, -5.03169584274292, -1.091538429260254, -0.5528503060340881, 0.03271952643990517, 0.0526634156703949, 0.016440190374851227, -0.02579418383538723, 0.10715527087450027, -0.009744048118591309],
    [0.02329482138156891, 0.21933576464653015, 0.040267977863550186, -0.07655417919158936, 1.7551501989364624, -0.014410154893994331, -0.19324982166290283, 0.20953533053398132, -0.20933137834072113, 0.1367415338754654, -0.0009180240449495614, 1.0963913202285767, -0.052944060415029526, 0.0769883394241333, -0.14415277540683746, -0.018233289942145348],
    [0.09159789234399796, 0.24272111058235168, 0.12052807211875916, -0.07679963856935501, 1.7147935628890991, -0.07336258143186569, -0.47304779291152954, 0.38181406259536743, -0.16268976032733917, 0.6770923733711243, 0.004319208208471537, 0.19864685833454132, 0.6627523899078369, -0.37509799003601074, -0.09607867896556854, -0.11855030804872513],
    [0.04805900901556015, 0.4691668450832367, -0.15610577166080475, -0.31037014722824097, 1.277097225189209, -0.13563401997089386, -0.10376353561878204, -0.052235428243875504, -0.2794184386730194, 0.8097546100616455, -0.1195620745420456, 0.006817629095166922, 0.6116837859153748, 0.41391894221305847, 0.3413048982620239, 0.45148342847824097],
    [0.15144023299217224, 0.22470158338546753, 0.3836003243923187, 0.42706066370010376, 0.4640190005302429, -0.08552108705043793, 0.12046020478010178, 0.0021140126045793295, -0.28891855478286743, 0.03941412270069122, 0.2768874168395996, -0.0884568989276886, -0.261484831571579, -0.6050072908401489, -0.2199770212173462, 0.6537340879440308],
    [0.40283897519111633, 0.17081186175346375, 0.33529049158096313, -0.35768941044807434, 0.870553195476532, -0.42392316460609436, 0.7650881409645081, -0.7119308114051819, -0.0649804174900055, -0.6493614315986633, 0.2504397928714752, 0.01723763346672058, -0.01888456381857395, -0.1592029482126236, 0.3550436794757843, -0.19347253441810608],
    [0.26571419835090637, 0.6518830060958862, 0.38408952951431274, 0.3787323534488678, 0.1794251799583435, -0.6631624698638916, 0.3805694580078125, -0.16141794621944427, -0.723532497882843, -0.19514645636081696, -1.0929720401763916, 1.2795950174331665, -0.2241012454032898, -0.5291746854782104, 0.1376325488090515, 0.22866342961788177],
])

# Shape (16, 1), from model.rnn.weight_ih_l0.
M_16_10_W_hi = np.array([
    [0.06919536739587784],
    [-10.564790725708008],
    [0.014607131481170654],
    [-0.12265294790267944],
    [10.156975746154785],
    [-0.28130465745925903],
    [-10.995721817016602],
    [-13.16869068145752],
    [-12.312334060668945],
    [-0.22756263613700867],
    [0.05984975025057793],
    [0.14799705147743225],
    [0.29800716042518616],
    [-0.04701320827007294],
    [-1.3206626176834106],
    [-0.1191084161400795],
])

# Shape (16,), the weight vector of the linear probe.
M_16_10_PROBE = np.array([
    0.09444186163805308,
    -0.7455110259093655,
    0.01304305582993644,
    0.8545640623647488,
    3.678650892008945,
    -0.045853798041207865,
    -0.7562236969871922,
    -1.0624333470795098,
    -0.6929935339358955,
    -3.131283866278705,
    0.0716980552584158,
    0.129229610859309,
    0.2600607835880996,
    -0.030293138139114016,
    -0.6733422825122595,
    -0.8279930076631422,
])

# figures_16_10.py substitutes this for the probe threshold, because the probe is
# fitted with a hard-coded intercept of 0.0 and the region would otherwise be
# degenerate.
TIIINY = 0.0001


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
def test_seqlen_2_rnn(trunc_dim, expected_layer_1_regions, expected_layer_0_regions):
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
def test_post_image_feasibility_is_monotone_at_layer_1(trunc_dim):
    """W_hh_hi has full row rank. The post image test is monotone here."""
    _, W_hh_hi, no_bias, root, _ = _pipeline(trunc_dim)

    feasible = _post_image_feasible_masks(root, W_hh_hi, no_bias, trunc_dim)

    # Both bounds matter. If every orthant qualified, or none did, up closure would
    # hold for trivial reasons and this test would assert nothing.
    assert 0 < len(feasible) < 2 ** trunc_dim
    assert _up_closure_violations(feasible, trunc_dim) == []


@pytest.mark.parametrize('trunc_dim', [2, 3, 4])
def test_post_image_feasibility_is_monotone_at_layer_0(trunc_dim):
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
def test_reverse_relu_with_pruning_matches_exhaustive_at_layer_1(trunc_dim):
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
def test_reverse_relu_with_pruning_matches_exhaustive_at_layer_0(trunc_dim):
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
# Timing, run from the terminal rather than under pytest
# ---------------------------------------------------------------------------


def _print_timing_row(
        trunc_dim, layer, exhaustive_seconds, pruned_seconds, regimes, calls, regions):
    speedup = exhaustive_seconds / pruned_seconds if pruned_seconds else float('nan')
    print(f'{trunc_dim:>5} {layer:>5} {exhaustive_seconds:>13.4f} {pruned_seconds:>10.4f} '
          f'{speedup:>7.2f}x {regimes:>8} {calls:>7} {regions:>8}')


def timing_report(trunc_dims=(9, 10), skip_exhaustive=False):
    """Print exhaustive against pruned wall clock, taken from return_metadata.

    This asserts nothing and pytest does not collect it, because its numbers depend
    on the machine and on what else is running. The times are the 'total_time_taken'
    the two methods report, which covers the orthant loop only and excludes setup.
    Layer 0 is summed over every region that layer 1 produced.

    'regions' counts the calls that produced a region rather than None. Pruned and
    exhaustive return the same regions, so it is read off the pruned run, which is
    the one that always happens.
    """
    print(f'{"trunc":>5} {"layer":>5} {"exhaustive s":>13} {"pruned s":>10} '
          f'{"speedup":>8} {"regimes":>8} {"calls":>7} {"regions":>8}')

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

        exhaustive_seconds = pruned_seconds = 0.0
        regimes = calls = regions = 0
        for region in layer_1_regions:
            if not skip_exhaustive:
                _, exhaustive = region.reverse_relu(W_hi, no_bias, return_metadata=True)
                exhaustive_seconds += exhaustive['total_time_taken']
            _, pruned = region.reverse_relu_with_pruning(
                W_hi, no_bias, return_metadata=True
            )
            pruned_seconds += pruned['total_time_taken']
            regimes += pruned['num_total_regimes']
            calls += pruned['num_verifier_calls']
            regions += pruned['num_regions']
        _print_timing_row(
            trunc_dim, 0, exhaustive_seconds, pruned_seconds, regimes, calls, regions
        )


if __name__ == '__main__':
    timing_report()
