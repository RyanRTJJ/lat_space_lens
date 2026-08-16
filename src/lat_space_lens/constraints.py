"""
Desc:       Backpropping of feature constraints to input spaces.
Date:       2025, Dec 3 started
Author:     ryan.rtjj@gmail.com
"""
from itertools import combinations
from pathlib import Path
import time

import cvxpy as cp
import joblib
import numpy as np
from scipy.linalg import null_space


class ConstraintSet:
    """
    A Constraint Set is essentially a collection of linear inequalities.
    These linear inequalities are CONJUNCTIVE (i.e. ALL must be satisfied).
    I.E. They demarcate ONE contiguous region in space. Since the
    inequalities look like this in matrix form:
    A @ x >= | > b
    we represent them with self.A, self.b, and self.inequality
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
            '+=======================================+\n'

        return to_print

    def __init__(
            self,
            A: np.ndarray | None = None,
            b: np.ndarray | float | None = None,
            inequalities: list[str] | str | None = None,
            A_eq: np.ndarray | None = None,
            b_eq: np.ndarray | float | None = None,
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

    @staticmethod
    def power_set(n: int):
        elements = range(n)
        return [list(combo) for r in range(n + 1) for combo in combinations(elements, r)]

    def add_constraints(
            self,
            A: np.ndarray,
            b: np.ndarray,
            inequalities: list[str] | str | None = None,
            is_inequality: bool = True,
    ):
        A, b, inequalities = self._enforce_correct_shapes(
            A,
            b,
            inequalities,
            is_inequality=is_inequality
        )
        dim = A.shape[1]

        if is_inequality:
            if isinstance(self.A, np.ndarray):
                assert dim == self.A.shape[1], f'`A` dim {dim} does not match existing'
                self.A = np.hstack([self.A, A])
                self.b = np.concatenate([self.b, b])
                self.inequalities += inequalities
            else:
                self.A = A
                self.b = b
                self.inequalities = inequalities
        else:
            if isinstance(self.A_eq, np.ndarray):
                assert dim == self.A_eq.shape[1], f'`A_eq` dim {dim} does not match existing'
                self.A_eq = np.hstack([self.A_eq, A])
                self.b_eq = np.concatenate([self.b_eq, b])
            else:
                self.A = A
                self.b = b

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
        inequalities_to_keep = [inequalities[i] for i in max_b_idx]
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

        problem.solve()

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
        problem.solve()

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
                problem.solve()

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
            relu_dim: int,
            zeroed_dim_idxs: list[int],
            z_to_leq_zero_constraint: dict[int, str],
            A_eq: np.ndarray,
            b_eq: np.ndarray,
    ):
        """
        Helper function to compute the pre-relu image in a particular orthant
        described by zeroed_dim_idxs. Remember that relu is ReLU(W @ x + bias).

        @param zeroed_dim_idxs:             The orthant of interest is negative in these dims
        @param z_to_leq_zero_constraint:    A cached mapping of if the hyperplane W @ x + bias
                                            (varies in z) or (does not vary but is negative in z)
        @param A_eq:                        The normal form equation of W @ x + bias
        @param b_eq:                        The normal form equation of W @ x + bias

        @return:                            (ConstraintSet | None)
        """
        TOL = 1e-10

        Z = np.array(zeroed_dim_idxs)
        positive_dim_idxs = [i for i in range(relu_dim) if i not in zeroed_dim_idxs]
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
        if len(keep_row_idxs > 0):
            A_altered = A_altered[keep_row_idxs]
            b_altered = b_altered[keep_row_idxs]
        else:
            A_altered = np.empty(shape=(0, relu_dim))
            b_altered = np.empty(shape=(0,))

        # Then, we need to compute the unrelu-ing of this. This means combining:
        #   1. A_Z @ x <= 0 (dimensions Z must be negative)
        #   2. A_F @ x >= 0 (dimensions F are free to move; positive)
        #   3. A_altered @ x >= self.b (existing constraints, intersect with Z axial hyperplane)
        #   4. Must be on plane W @ x + bias. Assume bias = 0 for now
        if len(F) == 0:
            A_F = np.empty(shape=(0, relu_dim))
            b_F = np.empty(shape=(0,))
        else:
            A_F = np.eye(relu_dim)[F,:]
            b_F = np.zeros_like(F)
        inequalities_F = [ConstraintSet.GEQ] * len(F)

        if len(Z) == 0:
            A_Z = np.empty(shape=(0, relu_dim))
            b_Z = np.empty(shape=(0,))
        else:
            A_Z = -np.eye(relu_dim)[Z,:]
            b_Z = np.zeros_like(Z)
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
            b_eq=b_eq
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

    def reverse_relu(
            self,
            W: np.ndarray,
            bias: np.ndarray,
    ) -> dict[tuple, 'ConstraintSet']:
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
            # Find a particular solution to test for negativity
            x_particular = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]

        for z in range(d_large):
            if not isinstance(A_eq, np.ndarray):
                z_to_leq_zero_constraint[z] = ConstraintSet.UNCONSTRAINED
            elif np.all(np.abs(null_basis[z, :]) < TOL):
                if x_particular[z] > TOL:
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
            )
            if potential_region:
                zeroed_dim_idxs_to_constraint_sets[Z_tuple] = potential_region

        _PROFILE_end = time.time()
        _PROFILE_total_time = _PROFILE_end - _PROFILE_start
        print(f'Time taken: {_PROFILE_total_time}')
        print(f'Regions produced this set: {len(zeroed_dim_idxs_to_constraint_sets)} / {2 ** d_large}\n')
        return zeroed_dim_idxs_to_constraint_sets

    def reverse_add_relu(
            self,
            W: np.ndarray,
            M: np.ndarray,
            bias: np.ndarray,
    ) -> dict[tuple, 'ConstraintSet']:
        """
        Same as reverse_relu, but for a relu whose pre-activation is fed by TWO
        branches instead of one:

            p = W @ x + M @ y + bias        (pre-activation)
            z = ReLU(p)                     (post-activation)

        `y` is treated as FREE, and the answer is over the joint variable
        [x; y]. This is a relaxation if y is itself a function of x upstream.

        The orthant enumeration is completely untouched by the extra branch:
        it happens in p-space (width d_large), which does not know about x or
        y at all. The ONLY thing that changes is the equality constraint. The
        achievable p live on the affine set bias + span([W M]), so with
        N = null_space([W M].T):

            N.T @ p = N.T @ (W @ x + M @ y + bias) = N.T @ bias

        i.e. A_eq = N.T and b_eq = A_eq @ bias, same expression as the
        single-branch case with [W M] substituted for W. So we can defer to
        reverse_relu on the stacked matrix outright.

        @param W:       shape (d_large, d_x)
        @param M:       shape (d_large, d_y)

        @return:        dict of zeroed_dim_idxs to ConstraintSet, in p-space
                        (width d_large). Feed these to reverse_add_up_proj to
                        land in joint [x; y] space.
        """
        assert W.shape[0] == M.shape[0], \
            f'W and M must agree on d_large, got {W.shape[0]} and {M.shape[0]}'
        return self.reverse_relu(np.hstack([W, M]), bias)

    def reverse_up_proj(self, W: np.ndarray, bias: np.ndarray):
        """
        This is supposed to be easy (entirely linear). You have (math convention):
        -   post-down-proj:             h = W @ x + bias
        -   post-up-proj constraint:    A @ h > b
        -   post-up-proj eq constraint: A_eq @ h == b_eq

        =>  A @ W @ x + A @ bias > b
        =>  A @ W @ x > b - A @ bias

        =>  A_eq @ W @ x + A_eq @ bias == b_eq
        =>  A_eq @ W @ x == b_eq - A_eq @ bias

        The only gotcha is that when you go from high d back down to low d,
        you can drop the equality constraints, because the equality constraints
        are due to the assumption that the points are on this hyperplane.
        """
        assert isinstance(self.A, np.ndarray), 'No constraints to reverse'

        new_A = self.A @ W
        new_B = self.b - self.A @ bias

        new_A_eq = None
        new_b_eq = None

        # if (not drop_eq) and isinstance(self.A_eq, np.ndarray):
        #     new_A_eq = self.A_eq @ W
        #     new_b_eq = self.b_eq - self.A_eq @ bias

        return ConstraintSet(
            A=new_A,
            b=new_B,
            inequalities=self.inequalities,
            A_eq=new_A_eq,
            b_eq=new_b_eq,
        )

    def reverse_add_up_proj(self, W: np.ndarray, M: np.ndarray, bias: np.ndarray):
        """
        The substitution step for the two-branch case: takes constraints on the
        pre-activation p and rewrites them over the joint variable [x; y].

        -   pre-activation:     p = W @ x + M @ y + bias
        -   constraint on p:    A @ p > b

        =>  A @ (W @ x + M @ y) + A @ bias > b
        =>  A @ [W M] @ [x; y] > b - A @ bias

        which is reverse_up_proj on the stacked matrix. The resulting
        ConstraintSet has width d_x + d_y, with the x block FIRST.

        @param W:       shape (d_large, d_x)
        @param M:       shape (d_large, d_y)
        """
        assert W.shape[0] == M.shape[0], \
            f'W and M must agree on d_large, got {W.shape[0]} and {M.shape[0]}'
        return self.reverse_up_proj(np.hstack([W, M]), bias)

    def reverse_down_proj(self, W: np.ndarray, bias: np.ndarray) -> "ConstraintSet":
        """
        This is supposed to be easy (entirely linear). You have (math convention):
        -   post-down-proj:             h = W @ x + bias
        -   post-up-proj constraint:    A @ h > b
        -   post-up-proj eq constraint: A_eq @ h == b_eq

        =>  A @ W @ x + A @ bias > b
        =>  A @ W @ x > b - A @ bias

        =>  A_eq @ W @ x + A_eq @ bias == b_eq
        =>  A_eq @ W @ x == b_eq - A_eq @ bias

        The only gotcha is that when you go from low d back up to high d,
        you need to add back the equality constraints that describe the
        hyperplane (span(W.T)).

        NOTE: unless???
        """
        assert isinstance(self.A, np.ndarray), 'No constraints to reverse'

        new_A = self.A @ W
        new_b = self.b - self.A @ bias

        assert not isinstance(self.A_eq, np.ndarray), \
            'Not expecting existing equality constraints, but got ' + \
            f'A_eq: \n{self.A_eq}\nb_eq: \n{self.b_eq}'

        # Constraint 3: Have to lie on plane
        # Q = null_space(W)       # Since W is (d_small, d_large), this gives (d_large, k)
        # A_eq = Q.T
        # b_eq = np.zeros(shape=(A_eq.shape[0], )) # Q.T @ bias
        A_eq = None
        b_eq = None

        return ConstraintSet(
            A=new_A,
            b=new_b,
            inequalities=self.inequalities,
            A_eq=A_eq,
            b_eq=b_eq,
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
