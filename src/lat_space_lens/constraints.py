"""
Desc:       Backpropping of feature constraints to input spaces.
Date:       2025, Dec 3 started
Author:     ryan.rtjj@gmail.com
"""
from itertools import combinations
from pathlib import Path
import time
from typing import Literal, overload

import cvxpy as cp
import joblib
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import null_space


# cvxpy's default solver is accurate on these feasibility LPs but occasionally
# dies outright on a badly conditioned one -- the regions coming out of a deep
# reversal can have condition numbers around 1e16, since each step multiplies
# more weight matrices together. SCIPY (HiGHS) is a dedicated LP solver and was
# measured to agree with the default on every problem the default managed to
# solve, so it is the fallback.
#
# SCS is deliberately NOT here. Being first-order it is far too loose on these
# degenerate problems: on one such batch it called 654 of 1280 regions feasible
# where both CLARABEL and HiGHS said 118.
_SOLVER_FALLBACKS = tuple(
    solver for solver in ('SCIPY',) if solver in cp.installed_solvers()
)


def _solve_or_fall_back(problem: cp.Problem) -> None:
    """
    Solves in place, trying the fallback solvers if the default one errors out.
    Re-raises the original error if none of them get anywhere either.
    """
    try:
        problem.solve()
        return
    except cp.error.SolverError as default_solver_error:
        for solver in _SOLVER_FALLBACKS:
            try:
                problem.solve(solver=solver)
                return
            except cp.error.SolverError:
                continue
        raise default_solver_error


class ConstraintSet:
    """
    A Constraint Set is essentially a collection of linear inequalities.
    These linear inequalities are CONJUNCTIVE (i.e. ALL must be satisfied).
    I.E. They demarcate ONE contiguous region in space. Since the
    inequalities look like this in matrix form:
    A @ x >= | > b
    we represent them with self.A, self.b, and self.inequality

    BLOCK STRUCTURE (self.block_sizes)
    ----------------------------------
    Once a layer INJECTS a variable that did not come from earlier layers
    (see reverse_add_up_proj), the variable this constraint set is written over
    stops being a single homogeneous vector and becomes a concatenation:

        [ active | y_earliest | ... | y_latest ]

    self.block_sizes records those widths, and it always sums to
    self.A.shape[1]. The invariant is:

    -   block_sizes[0] is the ACTIVE block: the only part still being
        reverse-propagated. Every reverse_* method transforms this block and
        this block only.
    -   Everything after it is frozen injected input. It is carried through
        each reversal untouched, and each new injection inserts its block at
        position 1, so the suffix ends up ordered earliest-layer-first.

    With no injections this is just [A.shape[1]] and every method behaves
    exactly as it did before block_sizes existed.
    """
    GEQ = '>='
    GE = '>'
    UNSATISFIABLE = 'UNSATISFIABLE'
    UNCONSTRAINED = 'UNCONSTRAINED'
    SOLVABLE = 'SOLVABLE'
    RELU = 'RELU'

    def _enforce_correct_shapes(
            self,
            A: np.ndarray,
            b: np.ndarray | float,
            inequalities: list[str] | str | None = None,
            is_inequality: bool = True
    ):
        """
        Enforces:
        -   `A` into 2D matrix,
        -   `b` into a 1D matrix
        -   `inequalities` into a 1D list
        """
        # Type check: cannot be None
        assert isinstance(A, np.ndarray)
        assert isinstance(b, (np.ndarray, float))
        if is_inequality:
            assert isinstance(inequalities, (str, list))

        assert len(A.shape) in [1, 2]
        if isinstance(b, np.ndarray):
            assert len(b.shape) == 1

        if len(A.shape) == 1:
            A = A[None,:]

        if isinstance(b, float):
            b = np.array([b])

        if is_inequality and isinstance(inequalities, str):
            inequalities = [inequalities]

        # By now, A should be 2D, b should be 1D, and inequalities should be 1D
        num_constraints = A.shape[0]
        assert num_constraints == b.shape[0], \
            f'num_constraints == {num_constraints}, b.shape: {b.shape}'

        if is_inequality:
            assert len(inequalities) == num_constraints

        if is_inequality:
            return A, b, inequalities
        else:
            return A, b, None

    def __str__(self):
        def format_numpy_array(arr: np.ndarray):
            arr_str = str(arr)
            prefixed = '\n'.join('| ' + line for line in arr_str.split('\n'))
            return prefixed

        to_print = \
            '+=======================================+\n' + \
            f'| A: {self.A.shape}\n' + \
            '+------------------------\n' + \
            format_numpy_array(self.A) + '\n' \
            '+------------------------\n' + \
            f'| b: {self.b.shape}\n' + \
            '+------------------------\n' + \
            format_numpy_array(self.b) + '\n'

        if self.A_eq is not None:
            to_print += \
            '+------------------------\n' + \
            f'| A_eq: {self.A_eq.shape}\n' + \
            '+------------------------\n' + \
            format_numpy_array(self.A_eq) + '\n' \
            '+------------------------\n' + \
            f'| b_eq: {self.b_eq.shape}\n' + \
            '+------------------------\n' + \
            format_numpy_array(self.b_eq) + '\n'

        to_print += \
            '+------------------------\n' + \
            f'| block_sizes: {self.block_sizes}\n' + \
            '+=======================================+\n'

        return to_print

    def __init__(
            self,
            A: np.ndarray | None = None,
            b: np.ndarray | float | None = None,
            inequalities: list[str] | str | None = None,
            A_eq: np.ndarray | None = None,
            b_eq: np.ndarray | float | None = None,
            block_sizes: list[int] | None = None,
    ):
        # Everything should be given or None
        assert (
            isinstance(A, np.ndarray) and \
            isinstance(b, (np.ndarray, float)) and \
            isinstance(inequalities, (list, str))
        ) or (
            not isinstance(A, np.ndarray) and \
            not isinstance(b, (np.ndarray, float)) and \
            not isinstance(inequalities, (list, str))
        )

        if isinstance(A, np.ndarray):
            A, b, inequalities = self._enforce_correct_shapes(A, b, inequalities)

        self.A = A      # This is 2D. Even if shape[0] == 1, it'd be (1, dim)
        self.b = b      # This is 1D. Even if shape[0] == 1, it'd be (1,)
        self.inequalities = inequalities

        # Same thing for equalities
        assert (
            isinstance(A_eq, np.ndarray) and \
            isinstance(b_eq, (np.ndarray, float))
        ) or (
            not isinstance(A_eq, np.ndarray) and \
            not isinstance(b_eq, (np.ndarray, float))
        )

        if isinstance(A_eq, np.ndarray):
            A_eq, b_eq, _ = self._enforce_correct_shapes(A_eq, b_eq, is_inequality=False)

        self.A_eq = A_eq
        self.b_eq = b_eq

        # A_eq is applied to the same variable as A (see is_feasible, which
        # builds ONE cp.Variable for both), so a width mismatch is a bug that
        # would otherwise only surface at solve time.
        if isinstance(self.A, np.ndarray) and isinstance(self.A_eq, np.ndarray):
            assert self.A_eq.shape[1] == self.A.shape[1], \
                f'A_eq width {self.A_eq.shape[1]} does not match A width {self.A.shape[1]}'

        if isinstance(self.A, np.ndarray):
            if block_sizes is None:
                block_sizes = [self.A.shape[1]]
            assert all(size > 0 for size in block_sizes), \
                f'block_sizes must all be positive, got {block_sizes}'
            assert sum(block_sizes) == self.A.shape[1], \
                f'block_sizes {block_sizes} must sum to A width {self.A.shape[1]}'
            self.block_sizes = list(block_sizes)
        else:
            assert block_sizes is None, 'block_sizes given but there are no constraints'
            self.block_sizes = None

    def __setstate__(self, state):
        """
        Back-compat for ConstraintSets pickled before block_sizes existed
        (save_to_pkl / load_from_pkl, and anything already sitting on disk or
        in flight between machines). An un-blocked set is a single block.
        """
        self.__dict__.update(state)
        if 'block_sizes' not in state:
            self.block_sizes = \
                [self.A.shape[1]] if isinstance(self.A, np.ndarray) else None

    @property
    def d_active(self) -> int:
        """
        Width of the block still being reverse-propagated.
        """
        assert self.block_sizes is not None, 'No constraints defined'
        return self.block_sizes[0]

    @property
    def d_suffix(self) -> int:
        """
        Total width of the frozen injected blocks trailing the active one.
        """
        return sum(self.block_sizes[1:])

    def _carry_suffix(self, W: np.ndarray, bias: np.ndarray):
        """
        Lifts a map on the ACTIVE block alone to a map on the whole variable,
        leaving the injected suffix untouched:

            [active; y] = [[W, 0], [0, I]] @ [new_active; y] + [bias; 0]

        so that self.A @ W_full == [A_active @ W | A_y], which is exactly the
        block-wise substitution. With no suffix this returns W and bias
        unchanged.
        """
        assert W.shape[0] == self.d_active, \
            f'W maps from the active block, so expected {self.d_active} rows, got {W.shape[0]}'
        assert bias.shape[0] == self.d_active, \
            f'bias applies to the active block, so expected {self.d_active}, got {bias.shape[0]}'

        d_suffix = self.d_suffix
        if d_suffix == 0:
            return W, bias

        W_full = block_diag(W, np.eye(d_suffix))
        bias_full = np.concatenate([bias, np.zeros(d_suffix)])
        return W_full, bias_full

    @staticmethod
    def power_set(n: int):
        elements = range(n)
        return [list(combo) for r in range(n + 1) for combo in combinations(elements, r)]

    @staticmethod
    def dedupe_constraints(A, b, inequalities):
        """
        Only for linear inequality constraints
        """
        assert isinstance(A, np.ndarray) and len(A.shape) == 2
        norms = np.linalg.norm(A, axis=1)
        # print(f'Deduping constraints. Pre-dedupe A: {A}, b: {b}')
        A_normalized = A / norms[:, None]
        b_normalized = b / norms

        # Rounding is for numerical stability reasons
        A_rounded = np.round(A_normalized, decimals=10)
        unique_rows, inverse_mapping = np.unique(A_rounded, axis=0, return_inverse=True)

        # Example:
        # A_rounded = np.array([
        #     [1.0, 0.0, 0.0],  # row 0
        #     [1.0, 0.0, 0.0],  # row 1
        #     [0.0, 1.0, 0.0]   # row 2
        # ])
        # unique_rows = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        # inverse_mapping = [1, 1, 0]

        max_b_idx = np.full(len(unique_rows), -1, dtype=int)
        max_b_val = np.full(len(unique_rows), -np.inf)

        for i, group in enumerate(inverse_mapping):
            if b_normalized[i] > max_b_val[group]:
                max_b_val[group] = b_normalized[i]
                max_b_idx[group] = i

        keep = sorted(max_b_idx)
        inequalities_to_keep = [inequalities[i] for i in keep]
        return A[keep], b[keep], inequalities_to_keep

    def is_feasible(self) -> bool:
        assert self.A is not None, 'No constraints defined'
        d_large = self.A.shape[1]

        x = cp.Variable(d_large)

        # Strict inequalities not allowed by cvxpy
        constraints = [self.A @ x >= self.b]

        # Add equality constraints if any
        if isinstance(self.A_eq, np.ndarray):
            constraints += [self.A_eq @ x == self.b_eq]

        # Dummy objective: minimize zero
        problem = cp.Problem(cp.Minimize(0), constraints)

        _solve_or_fall_back(problem)

        # Check feasibility
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return True
        else:
            return False

    def has_area(self) -> bool:
        """
        Tests if all the linear inequalities are strictly satisfiable by:

        mazimie delta
        subject to
            A @ x >= b_ineq + delta
            A_eq @ x == b_eq
        """
        _, dim = self.A.shape

        x = cp.Variable(dim)
        delta = cp.Variable()

        constraints = [self.A @ x >= self.b + delta]

        if isinstance(self.A_eq, np.ndarray):
            constraints += [self.A_eq @ x == self.b_eq]

        problem = cp.Problem(cp.Maximize(delta), constraints)
        _solve_or_fall_back(problem)

        if problem.status == cp.UNBOUNDED:
            # Infinite, basically
            return True
        elif problem.status == cp.OPTIMAL:
            return delta.value > 1e-6
        else:
            return False

    def remove_redundant_constraints(self):
        """
        Only for linear inequalities.
        NOTE: In-place modification of self.A, self.b, and self.inequalities!
        NOTE: Kinda slow and buggy.

        This is a greedy algorithm (O(num_constraints**2)) where
        num_constraints ~= dim
        """
        assert self.A is not None, 'No constraints defined'
        num_constraints, d = self.A.shape
        x = cp.Variable(d)

        if isinstance(self.A_eq, np.ndarray):
            equality_constraints = [self.A_eq @ x == self.b_eq]
        else:
            equality_constraints = []

        # print('\nRemoving redundant constraints...')
        # print(f'pre: {self}')

        made_change = True
        kept_indices = list(range(num_constraints))

        while made_change:
            made_change = False
            for idx, i in enumerate(kept_indices):
                constraints = equality_constraints.copy()
                for j in kept_indices:
                    if j != i:
                        constraints += [self.A[j] @ x >= self.b[j]]

                # Check if constraint i is redundant
                objective = cp.Minimize(self.A[i] @ x - self.b[i])
                problem = cp.Problem(objective, constraints)
                _solve_or_fall_back(problem)

                # If minimum value >= -epsilon, constraint i is redundant
                if problem.status == cp.OPTIMAL and problem.value >= -1e-6:
                    kept_indices.pop(idx)
                    made_change = True
                    break

        kept_indices_np = np.array(kept_indices)
        self.A = self.A[kept_indices_np]
        self.b = self.b[kept_indices_np]
        self.inequalities = [self.inequalities[i] for i in kept_indices]

        # print(f'post: {self}')

    @staticmethod
    def flatten_region_dicts(
            Z_to_regions: dict[tuple, 'ConstraintSet'],
    ) -> list['ConstraintSet']:
        """
        Syntactic sugar, really, or you could just call .values() on the dict XD
        """
        return list(Z_to_regions.values())

    def reverse_relu_for_Z_tuple(
            self,
            d_active: int,
            zeroed_dim_idxs: list[int],
            z_to_leq_zero_constraint: dict[int, str],
            A_eq: np.ndarray,
            b_eq: np.ndarray,
            total_width: int | None = None,
    ):
        """
        Helper function to compute the pre-relu image in a particular orthant
        described by zeroed_dim_idxs. Remember that relu is ReLU(W @ x + bias).

        The relu acts on the ACTIVE block only, so the orthant enumeration runs
        over the first d_active dims while every constraint is still written at
        full width. Building the cone constraints from np.eye(total_width) and
        selecting rows from the active block gives the zero-padding across the
        injected suffix for free.

        @param d_active:                    Width of the block the relu acts on
        @param zeroed_dim_idxs:             The orthant of interest is negative in these dims.
                                            Indices into the ACTIVE block.
        @param z_to_leq_zero_constraint:    A cached mapping of if the hyperplane W @ x + bias
                                            (varies in z) or (does not vary but is negative in z)
        @param A_eq:                        The normal form equation of W @ x + bias, already
                                            padded out to total_width
        @param b_eq:                        The normal form equation of W @ x + bias
        @param total_width:                 Full variable width. Defaults to d_active, i.e. the
                                            un-blocked case.

        @return:                            (ConstraintSet | None)
        """
        TOL = 1e-10

        if total_width is None:
            total_width = d_active

        Z = np.array(zeroed_dim_idxs)
        positive_dim_idxs = [i for i in range(d_active) if i not in zeroed_dim_idxs]
        F = np.array(positive_dim_idxs)

        # Does the plane even have a satisfiable region?
        can_unrelu = True
        for z in Z:
            if z_to_leq_zero_constraint[z] == ConstraintSet.UNSATISFIABLE:
                # This pretty much never happens
                print('Unsatisfiable')
                can_unrelu = False
                break
        if not can_unrelu:
            return None

        # First, we need to get the intersection of self.A and A_Z @ x == 0.
        # This means:
        #   1. A_altered = A.copy(), A_altered[:, Z] = 0
        #   2. Combine that wth A_Z @ x == 0
        #   3. Checking for feasibility
        A_altered = self.A.copy()
        b_altered = self.b.copy()
        if len(Z) > 0:
            A_altered[:, Z] = 0
        row_norms = np.linalg.norm(A_altered, axis=1)
        infeasible = (row_norms < TOL) & (self.b > TOL) # This is correct
        if np.any(infeasible):
            # This pretty much never happens except at the very last layer
            return None

        # Delete all null rows
        null_row_idxs = np.argwhere(row_norms < TOL)
        keep_row_idxs = np.array([i for i in range(len(A_altered)) if i not in null_row_idxs])
        if len(keep_row_idxs) > 0:
            A_altered = A_altered[keep_row_idxs]
            b_altered = b_altered[keep_row_idxs]
        else:
            A_altered = np.empty(shape=(0, total_width))
            b_altered = np.empty(shape=(0,))

        # Then, we need to compute the unrelu-ing of this. This means combining:
        #   1. A_Z @ x <= 0 (dimensions Z must be negative)
        #   2. A_F @ x >= 0 (dimensions F are free to move; positive)
        #   3. A_altered @ x >= self.b (existing constraints, intersect with Z axial hyperplane)
        #   4. Must be on plane W @ x + bias. Assume bias = 0 for now
        if len(F) == 0:
            A_F = np.empty(shape=(0, total_width))
            b_F = np.empty(shape=(0,))
        else:
            A_F = np.eye(total_width)[F,:]
            b_F = np.zeros(len(F))
        inequalities_F = [ConstraintSet.GEQ] * len(F)

        if len(Z) == 0:
            A_Z = np.empty(shape=(0, total_width))
            b_Z = np.empty(shape=(0,))
        else:
            A_Z = -np.eye(total_width)[Z,:]
            b_Z = np.zeros(len(Z))
        inequalities_Z = [ConstraintSet.GEQ] * len(Z)

        A_combined = np.vstack([
            A_F,
            A_Z,
            A_altered
        ])
        b_combined = np.hstack([b_F, b_Z, b_altered])
        inequalities_combined = inequalities_F + inequalities_Z + self.inequalities

        A_deduped, b_deduped, inequalities_deduped = ConstraintSet.dedupe_constraints(
            A_combined,
            b_combined,
            inequalities_combined
        )

        region = ConstraintSet(
            A=A_deduped,
            b=b_deduped,
            inequalities=inequalities_deduped,
            A_eq=A_eq,
            b_eq=b_eq,
            # Still in pre-activation space: same widths, same layout
            block_sizes=self.block_sizes,
        )

        # It could be unfeasible now even if the hyperplane passed the
        # previous solvability check, because of the additional Free cone
        # constraints.
        # TODO: Here is probably the only place for mathematical
        # optimizations
        if region.is_feasible():
            # print('feasible\n')

            # This shit is really slow (takes ~0.01 ~ 0.1 seconds for d_large=15)
            # region.remove_redundant_constraints()

            return region
        else:
            # Empty region, dont add
            # print('infeasible\n')
            return None

    @overload
    def reverse_relu(
            self,
            W: np.ndarray,
            bias: np.ndarray,
            return_metadata: Literal[False] = False,
    ) -> dict[tuple, 'ConstraintSet']: ...

    @overload
    def reverse_relu(
            self,
            W: np.ndarray,
            bias: np.ndarray,
            return_metadata: Literal[True],
    ) -> tuple[dict[tuple, 'ConstraintSet'], dict]: ...

    def reverse_relu(
            self,
            W: np.ndarray,
            bias: np.ndarray,
            return_metadata: bool = False,
    ) -> dict[tuple, 'ConstraintSet'] | tuple[dict[tuple, 'ConstraintSet'], dict]:
        """
        NOTE:           This function is to be run when you can run reverse_relu on a single
                        machine only. Usually this means that relu_dim is <= 16. For parallel
                        reverse_relu, map reverse_relu_for_Z_tuple for all Z_tuples over
                        multiple cores / machines. View reverse_relu_orchestrator.py and
                        worker.py as an example.

        You have (math convention):
        -   post-up-proj + relu:        h = ReLU(W @ x + b)
        -   post-up-proj constraint:    A @ h > b

        @param W:       shape (big, small) the up_projection matrix

        @return:        We return a dict of zeroed_dim_idxs to ConstraintSet, mainly for
                        debugging purposes. You can call flatten_region_dicts() to turn this
                        into a list afterwards.
        """
        TOL = 1e-10
        assert isinstance(self.A, np.ndarray), 'Cant call reverse_up_proj_relu on no constraints'

        # Possible optimization: look for constraints that already look like relu constraints
        # and dedupe them. Don't think there'll be very many though.
        d_large, _ = W.shape
        assert bias.shape[0] == d_large, 'Check your W and bias dims. They need to match.'

        # The relu acts on the active block only; the injected suffix rides along.
        total_width = self.A.shape[1]
        assert d_large == self.d_active, \
            f'relu acts on the active block, so expected W to have {self.d_active} rows, ' \
            f'got {d_large}'

        dim_combos_for_zeroed_surfaces = self.power_set(d_large)

        # First, figure out which dimensions in which W is being meaningfully unrelued
        # (i.e. exists region of hyperplane that is negative in that direction)
        # dimension-wise, we wanna see if the hyperplane varies in that dim
        # NOTE: This can be cached
        z_to_leq_zero_constraint = {}
        null_basis = null_space(W.T)        # (d_large, rank of null space)

        # Precompute the equation of plane in normal form too
        A_eq = null_basis.T
        b_eq = A_eq @ bias
        if len(A_eq) == 0:
            # Happens when the relu is following a down-projection
            A_eq = None
            b_eq = None
        else:
            # Find a particular solution to test for negativity. Do this BEFORE
            # padding, while A_eq is still square with the relu's own dims.
            # x_particular = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]

            # The equation of the plane constrains the active block only, so
            # widen it with zeros over the injected suffix. b_eq is unchanged.
            if total_width > d_large:
                A_eq = np.hstack([A_eq, np.zeros((A_eq.shape[0], total_width - d_large))])

        for z in range(d_large):
            if not isinstance(A_eq, np.ndarray):
                z_to_leq_zero_constraint[z] = ConstraintSet.UNCONSTRAINED
            elif np.all(np.abs(W[z, :]) < TOL):
                # if x_particular[z] > TOL:
                if bias[z] > 0:
                    # This hyperplane is at dim[z] > 0
                    z_to_leq_zero_constraint[z] = ConstraintSet.UNSATISFIABLE
                else:
                    # The entire hyperplane is <= 0
                    z_to_leq_zero_constraint[z] = ConstraintSet.UNCONSTRAINED
            else:
                z_to_leq_zero_constraint[z] = ConstraintSet.SOLVABLE

        zeroed_dim_idxs_to_constraint_sets = {}

        _PROFILE_start = time.time()

        for zeroed_dim_idxs in dim_combos_for_zeroed_surfaces:
            Z_tuple = tuple(sorted(zeroed_dim_idxs))
            potential_region = self.reverse_relu_for_Z_tuple(
                d_large,
                zeroed_dim_idxs,
                z_to_leq_zero_constraint,
                A_eq,
                b_eq,
                total_width=total_width,
            )
            if potential_region:
                zeroed_dim_idxs_to_constraint_sets[Z_tuple] = potential_region

        _PROFILE_end = time.time()
        _PROFILE_total_time = _PROFILE_end - _PROFILE_start
        _metadata = {}
        _metadata['total_time_taken'] = _PROFILE_total_time
        _metadata['num_regions'] = len(zeroed_dim_idxs_to_constraint_sets)
        _metadata['num_total_regimes'] = 2 ** d_large

        if return_metadata:
            return zeroed_dim_idxs_to_constraint_sets, _metadata
        return zeroed_dim_idxs_to_constraint_sets

    def reverse_up_proj(
            self,
            W: np.ndarray,
            bias: np.ndarray,
            layer_is_followed_by_relu: bool = True,
    ):
        """
        This is supposed to be easy (entirely linear). You have (math convention):
        -   post-up-proj:               h = W @ x + bias
        -   post-up-proj constraint:    A @ h > b
        -   post-up-proj eq constraint: A_eq @ h == b_eq

        =>  A @ W @ x + A @ bias > b
        =>  A @ W @ x > b - A @ bias

        =>  A_eq @ W @ x + A_eq @ bias == b_eq
        =>  A_eq @ W @ x == b_eq - A_eq @ bias

        @param layer_is_followed_by_relu:
            True (default) means A_eq is the image constraint reverse_relu built
            from this same W: A_eq == null_space(W.T).T, b_eq == A_eq @ bias.
            Then A_eq @ W == 0 and b_eq - A_eq @ bias == 0, so the substitution
            leaves 0 == 0 and we drop it. Dropped rather than computed because
            A_eq @ W comes out as ~1e-16 noise, not exact zeros.

            False means A_eq is some other equality, substituted through like
            the inequality block. If the result is contradictory the region is
            empty and is_feasible() says so.
        """
        assert isinstance(self.A, np.ndarray), 'No constraints to reverse'

        W_full, bias_full = self._carry_suffix(W, bias)

        new_A = self.A @ W_full
        new_B = self.b - self.A @ bias_full

        if layer_is_followed_by_relu or not isinstance(self.A_eq, np.ndarray):
            new_A_eq = None
            new_b_eq = None
        else:
            new_A_eq = self.A_eq @ W_full
            new_b_eq = self.b_eq - self.A_eq @ bias_full

        return ConstraintSet(
            A=new_A,
            b=new_B,
            inequalities=self.inequalities,
            A_eq=new_A_eq,
            b_eq=new_b_eq,
            block_sizes=[W.shape[1]] + self.block_sizes[1:],
        )

    def reverse_add_up_proj(
            self,
            W: np.ndarray,
            M: np.ndarray,
            bias: np.ndarray,
            layer_is_followed_by_relu: bool = True,
    ):
        """
        The substitution step for the two-branch case: takes constraints on the
        pre-activation p and rewrites them over the joint variable [x; y].

        -   pre-activation:     p = W @ x + M @ y + bias
        -   constraint on p:    A @ p > b

        =>  A @ (W @ x + M @ y) + A @ bias > b
        =>  A @ [W M] @ [x; y] > b - A @ bias

        which is reverse_up_proj on the stacked matrix. The active block, width
        d_large, is replaced by TWO blocks, [d_x, d_y], with the x block first;
        any already-injected suffix is carried along behind them.

        The new y block is inserted at position 1 rather than appended, so that
        walking further back through the network leaves the suffix ordered
        earliest-injecting-layer first:

            [u, y_earliest, ..., y_latest]

        @param W:       shape (d_large, d_x)
        @param M:       shape (d_large, d_y)
        @param layer_is_followed_by_relu:
                        Forwarded to reverse_up_proj; see there.
        """
        assert W.shape[0] == M.shape[0], \
            f'W and M must agree on d_large, got {W.shape[0]} and {M.shape[0]}'

        carried_blocks = self.block_sizes[1:]
        region = self.reverse_up_proj(
            np.hstack([W, M]),
            bias,
            layer_is_followed_by_relu=layer_is_followed_by_relu,
        )
        region.block_sizes = [W.shape[1], M.shape[1]] + carried_blocks
        return region

    def reverse_down_proj(
            self,
            W: np.ndarray,
            bias: np.ndarray,
            layer_is_followed_by_relu: bool = True,
    ) -> "ConstraintSet":
        """
        This is supposed to be easy (entirely linear). You have (math convention):
        -   post-down-proj:                 h = W @ x + bias
        -   post-down-proj constraint:      A @ h > b
        -   post-down-proj eq constraint:   A_eq @ h == b_eq

        =>  A @ W @ x + A @ bias > b
        =>  A @ W @ x > b - A @ bias

        =>  A_eq @ W @ x + A_eq @ bias == b_eq
        =>  A_eq @ W @ x == b_eq - A_eq @ bias

        Same arithmetic as reverse_up_proj; W is just wide here. Nothing is
        inverted -- this is the preimage, unbounded along null(W).

        No hyperplane is added going back up: a full-row-rank down-proj is
        surjective, so every x is reachable. Pinning x to span(W.T) would keep
        only the pseudo-inverse's solution and throw away the rest.

        @param layer_is_followed_by_relu:   See reverse_up_proj.
        """
        assert isinstance(self.A, np.ndarray), 'No constraints to reverse'

        W_full, bias_full = self._carry_suffix(W, bias)

        new_A = self.A @ W_full
        new_b = self.b - self.A @ bias_full

        if layer_is_followed_by_relu or not isinstance(self.A_eq, np.ndarray):
            A_eq = None
            b_eq = None
        else:
            A_eq = self.A_eq @ W_full
            b_eq = self.b_eq - self.A_eq @ bias_full

        return ConstraintSet(
            A=new_A,
            b=new_b,
            inequalities=self.inequalities,
            A_eq=A_eq,
            b_eq=b_eq,
            block_sizes=[W.shape[1]] + self.block_sizes[1:],
        )

    def save_to_pkl(self, file_path: str | Path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load_from_pkl(cls, file_path: str | Path):
        return joblib.load(file_path)


def reverse_relu_helper(_region: ConstraintSet, W, b):
    return ConstraintSet.flatten_region_dicts(
        _region.reverse_relu(W, b)
    )
