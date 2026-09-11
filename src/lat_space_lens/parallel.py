"""
Desc:       Running reverse_relu over many regions at once.
Date:       2026, Sep 10 started
Author:     ryan.rtjj@gmail.com

WHAT IS PARALLELISED, AND WHY IT IS THIS AND NOT THE ORTHANTS
-------------------------------------------------------------
The unit of work here is one whole call to ConstraintSet.reverse_relu_with_pruning
(or reverse_relu): one region, all of its orthants, on one worker.

The finer-grained split -- handing individual orthants, i.e. individual
reverse_relu_for_Z_tuple calls, to different workers -- looks more attractive
because the orthants outnumber the regions, but it fights the pruning. The pruned
search is stateful and greedy: `max_infeas` grows as orthants come back empty, and
every later orthant is tested against it before a linear program is ever built.
Sharing that state across processes means either paying for a lock and a round trip
on every orthant, or letting each worker keep its own copy, in which case the
workers re-derive prunes the others have already found and the search does more
linear programs than the serial one it is meant to be speeding up.

Regions carry no such shared state. Two regions' searches never look at each other,
so splitting on them is embarrassingly parallel and loses no pruning at all: each
worker runs the identical greedy search the serial code would have run for that
region. That is why the layer, not the orthant loop, is the thing spread out.

Executors hold their workers open across calls, since a deep reversal is one call
per layer and a pool per layer would pay the startup every time. Call close(), or
use the executor as a context manager, to hand them back; see Executor.close for
why that is worth doing before a script exits on CPython 3.12.

The one cost is that the regions in a layer are not equally expensive -- a region
whose post image dies early finishes in milliseconds while its neighbour walks
thousands of orthants -- so a static split of the list would leave workers idle.
The executors below therefore hand out small batches dynamically rather than
slicing the list into n_jobs contiguous pieces.
"""
from abc import ABC, abstractmethod
from contextlib import ExitStack
import atexit
from functools import partial
import os
import time

import joblib
import numpy as np

from .constraints import ConstraintSet


# The metadata fields reverse_relu and reverse_relu_with_pruning report that are
# counts of work done, and so aggregate over regions by summing. 'total_time_taken'
# is summed too but is handled separately, since summing seconds across workers
# gives CPU time rather than wall clock and the two need different names.
_METADATA_COUNTERS = (
    'num_regions',
    'num_total_regimes',
    'num_verifier_calls',
    'num_post_image_feasible',
    'num_maximal_infeasible',
)


def aggregate_metadata(metadatas, wall_clock_seconds=None) -> dict:
    """
    Fold the per-region metadata dicts of one layer into a single dict.

    The counters are summed. So are the seconds, but under the name
    'total_cpu_seconds', because a sum over workers that ran at the same time is
    no longer a duration anyone waited: it is the serial cost of the same work,
    which is exactly the number to compare a serial run against.

    reverse_relu reports fewer counters than reverse_relu_with_pruning does, so a
    counter is only present in the result if the parts had it, and all of them
    must agree -- a mixed batch is a caller bug, not something to paper over.

    @param metadatas:           the per-region dicts, in region order
    @param wall_clock_seconds:  the time the layer actually took, recorded by the
                                caller around the whole map. Stored as
                                'wall_clock_seconds', and left out when not given.
    @return:                    the aggregate, plus 'num_regions_in' for how many
                                regions went in
    """
    metadatas = list(metadatas)
    aggregate = {'num_regions_in': len(metadatas)}

    for counter in _METADATA_COUNTERS:
        present = [counter in metadata for metadata in metadatas]
        if not any(present):
            continue
        assert all(present), \
            f'{counter} is reported by some regions and not others, so the batch ' \
            f'mixes reverse_relu with reverse_relu_with_pruning'
        aggregate[counter] = sum(metadata[counter] for metadata in metadatas)

    aggregate['total_cpu_seconds'] = sum(
        metadata['total_time_taken'] for metadata in metadatas
    )
    if wall_clock_seconds is not None:
        aggregate['wall_clock_seconds'] = wall_clock_seconds
    return aggregate


class Executor(ABC):
    """
    How a layer's regions get spread out. The only thing the reversal asks of a
    backend is an order-preserving map, so that is the whole interface.

    Order-preserving matters: callers line the results up against the regions
    they came from, and the tests compare a parallel run against a serial one
    element by element.
    """

    @abstractmethod
    def map(self, function, items) -> list:
        """Apply `function` to every item, returning the results in input order."""

    @property
    def num_workers(self) -> int:
        """How many items can be in flight at once. 1 means no parallelism."""
        return 1

    def close(self) -> None:
        """
        Shut the workers down now, rather than whenever the interpreter gets round
        to it. Idempotent, and not final: a later map just starts them again.

        Worth calling at the end of a long script. Besides handing the cores back,
        it is the moment at which the workers are down and multiprocessing's
        resource tracker can be stopped in the open rather than in a finalizer,
        which is what keeps CPython 3.12 from printing an alarming
        `ChildProcessError: [Errno 10] No child processes` traceback as the
        interpreter exits. See stop_resource_tracker.
        """

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False


class SerialExecutor(Executor):
    """
    A plain loop. The default, and the reference the parallel executors are
    checked against: nothing about the search changes, so the regions come back
    identical rather than merely equivalent.
    """

    def map(self, function, items) -> list:
        return [function(item) for item in items]


class JoblibExecutor(Executor):
    """
    joblib.Parallel, one process per worker by default.

    @param n_jobs:                  Worker count. -1 means every core, as in
                                    joblib. 1 is a serial loop through joblib's
                                    own machinery, which is not quite
                                    SerialExecutor but is close enough to be
                                    useful when comparing overheads.
    @param backend:                 'loky' (default) gives real processes and is
                                    the only one that helps here: the orthant loop
                                    spends its time in cvxpy's Python-level
                                    problem building as much as in the solver, so
                                    'threading' would sit behind the GIL.
    @param batch_size:              How many regions a worker takes at a time.
                                    joblib's 'auto' grows the batch until each one
                                    takes a sensible amount of time, which is what
                                    keeps the uneven region costs balanced.
    @param inner_max_num_threads:   Threads each worker's BLAS and solver may use.
                                    Pinned to 1 by default, because the outer
                                    processes already own every core and letting
                                    each of them start a full thread pool oversubscribes
                                    the machine badly. Pass None to leave the
                                    worker's threading alone.
    @param verbose:                 Forwarded to joblib.Parallel.
    """

    def __init__(
            self,
            n_jobs: int = -1,
            backend: str = 'loky',
            batch_size='auto',
            inner_max_num_threads: int | None = 1,
            verbose: int = 0,
    ):
        self.n_jobs = n_jobs
        self.backend = backend
        self.batch_size = batch_size
        self.inner_max_num_threads = inner_max_num_threads
        self.verbose = verbose

        # The open pool, and the contexts holding it open, or None when there is
        # none up. Built on the first map and kept for the executor's lifetime:
        # a layer of a deep reversal is one map call, and spawning a fresh pool
        # for each of them would pay the startup on every layer.
        self._parallel = None
        self._stack = None

    @property
    def num_workers(self) -> int:
        if self.n_jobs < 0:
            # joblib's own convention: -1 is every cpu, -2 all but one, and so on.
            return max(1, (os.cpu_count() or 1) + 1 + self.n_jobs)
        return self.n_jobs

    def _ensure_pool(self) -> 'joblib.Parallel':
        """The open pool, starting one if this executor has none up."""
        if self._parallel is not None:
            return self._parallel

        config = {
            'backend': self.backend,
            'n_jobs': self.n_jobs,
        }
        # Only the process backends know what to do with a nested thread cap, and
        # joblib raises rather than ignoring it on the ones that do not.
        if self.inner_max_num_threads is not None and self.backend in (
                'loky', 'multiprocessing'):
            config['inner_max_num_threads'] = self.inner_max_num_threads

        stack = ExitStack()
        try:
            stack.enter_context(joblib.parallel_config(**config))
            self._parallel = stack.enter_context(joblib.Parallel(
                batch_size=self.batch_size,
                verbose=self.verbose,
            ))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self._parallel

    def map(self, function, items) -> list:
        parallel = self._ensure_pool()
        # Wrapped so the worker running it registers its own tracker cleanup; see
        # _TaskWithTrackerCleanup.
        work = _TaskWithTrackerCleanup(function)
        return list(parallel(
            joblib.delayed(work)(item) for item in items
        ))

    def close(self) -> None:
        """
        Shut the pool down now. A later map starts a fresh one.

        Letting go of joblib's Parallel is not enough on its own: joblib keeps the
        worker processes of its loky backend warm afterwards, deliberately, so
        that the next parallel call anywhere in the program does not pay to spawn
        them. Those still-running workers are exactly what this is meant to take
        down, so it goes on to call shutdown_workers. That pool is shared process
        wide, so do not close an executor while other joblib work is in flight.
        """
        stack, self._stack, self._parallel = self._stack, None, None
        if stack is not None:
            stack.close()
        shutdown_workers()


def resolve_executor(executor) -> Executor:
    """
    Turn the `executor` argument of the functions below into an Executor.

    None or 1 means serial, an int means that many joblib workers, and an
    Executor is itself. Callers therefore never have to import anything from here
    to ask for parallelism, and `executor=1` in a test means literally the serial
    code path rather than a one-worker process pool.
    """
    if executor is None or executor == 1:
        return SerialExecutor()
    if isinstance(executor, Executor):
        return executor
    if isinstance(executor, (int, np.integer)):
        return JoblibExecutor(n_jobs=int(executor))
    raise TypeError(
        f'executor must be None, an int, or an Executor, got {type(executor)}'
    )


# How long stop_resource_tracker waits for the tracker child to go away. It exits
# as soon as the last write end of its pipe is closed, so this is a bound on a
# wait that normally ends at once, not a duration anyone should expect to spend.
_TRACKER_EXIT_TIMEOUT_SECONDS = 5.


def stop_resource_tracker(timeout: float = _TRACKER_EXIT_TIMEOUT_SECONDS) -> bool:
    """
    Stop multiprocessing's resource tracker, here, where an error is catchable.

    This is what silences

        Exception ignored in: <function ResourceTracker.__del__ ...>
        ...
        File ".../multiprocessing/resource_tracker.py", line 111, in _stop_locked
        ChildProcessError: [Errno 10] No child processes

    which CPython 3.12 prints at interpreter shutdown from the finalizer it gave
    ResourceTracker in python/cpython#88887. The finalizer waits on the tracker
    child; when that child has already been reaped the wait raises, inside a
    finalizer, so the interpreter prints the traceback and moves on. The exception
    is swallowed, the exit status is untouched, and it is printed after every
    result has been returned -- it is noise, but it is noise that no try block of
    the caller's can reach, because of where it is raised.

    So the tracker is stopped HERE instead, at a moment of our choosing, doing
    what the finalizer would have done later: close the pipe, which is what tells
    the tracker to exit, then reap it. Any ChildProcessError lands in this
    function, where catching it is ordinary. Afterwards the tracker's _fd and _pid
    are None, and the finalizer's first two lines return on exactly that, so the
    traceback cannot happen. Nothing is patched and nothing is permanently gone:
    the next piece of code that needs a tracker calls ensure_running and gets a
    fresh one, which is the same path a tracker that had exited on its own takes.

    CALL THIS ONLY WITH THE WORKER POOL ALREADY DOWN. Every worker holds a
    duplicate of the tracker pipe's write end, and the tracker does not exit while
    any of them is open, so a blocking wait with workers still up never returns --
    which is why this waits with a deadline and its own WNOHANG loop rather than
    calling the tracker's _stop. On timeout it gives up on reaping but still clears
    _pid, so the child is left to the process exit that would have collected it
    anyway, and the finalizer still returns early.

    Everything here is private to multiprocessing, so every step is guarded and a
    version that has moved on just leaves this doing nothing.

    @return:    True if a running tracker was stopped, False if there was none or
                if the internals were not as expected
    """
    from multiprocessing import resource_tracker

    tracker = getattr(resource_tracker, '_resource_tracker', None)
    lock = getattr(tracker, '_lock', None)
    if lock is None:
        return False

    try:
        with lock:
            fd = tracker._fd
            pid = tracker._pid
            if fd is None or pid is None:
                # Never started here, or already stopped. Either way the finalizer
                # returns early and there is nothing to do.
                return False

            # Closing the write end is what stops the tracker's main loop; this is
            # the tracker's own mechanism, not a signal.
            os.close(fd)
            tracker._fd = None

            deadline = time.monotonic() + timeout
            while True:
                try:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    # Somebody else got there first. This is the very case the
                    # finalizer cannot survive, and here it is just a branch.
                    break
                if reaped == pid or time.monotonic() >= deadline:
                    break
                time.sleep(0.01)

            tracker._pid = None
            return True
    except Exception:
        # The internals are not what this expects, so leave the tracker alone. The
        # cost is the shutdown traceback coming back, which harms nothing.
        return False


class _TaskWithTrackerCleanup:
    """
    A task, carrying the same cleanup into the worker that runs it.

    A worker usually inherits the parent's tracker pipe and never starts a tracker
    of its own, in which case stop_resource_tracker finds no _pid and does nothing.
    A worker that did start one would otherwise print the same traceback as it
    exits, and unlike the parent it has no natural place for a caller to intervene,
    so the hook is registered on its way through the first task.

    A class at module scope rather than a closure, so that it pickles. Registration
    happens once per worker; after that this is a flag check in front of the call.
    """

    def __init__(self, function):
        self.function = function

    def __call__(self, item):
        global _worker_cleanup_registered
        if not _worker_cleanup_registered:
            _worker_cleanup_registered = True
            # atexit rather than a finalizer: it runs while the interpreter is
            # still whole, before the finalizer that would raise.
            atexit.register(stop_resource_tracker)
        return self.function(item)


# Whether this process has already registered the hook above. Module state because
# a worker runs many tasks and only the first needs to do it.
_worker_cleanup_registered = False


def shutdown_workers() -> None:
    """
    Shut down the worker processes joblib keeps warm between calls.

    Closing an Executor is enough when you own it, but `executor=8` builds a
    throwaway one per call and the pool it used stays up on purpose, so that the
    next call does not pay to spawn it again. That is the pool still standing when
    a script ends. This takes it down.

    Safe to call at any time and however many times: joblib spawns a new pool on
    the next parallel call. Worth calling before a long script exits, for the
    reason given in Executor.close.

    Once the workers are down this also stops multiprocessing's resource tracker,
    which is the step that keeps CPython 3.12 from printing an alarming
    ChildProcessError traceback out of a finalizer as the interpreter exits. The
    order matters, which is why the two are in the one function; see
    stop_resource_tracker.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
    except ImportError:
        # No loky, so there is no pool of its to take down.
        return
    get_reusable_executor().shutdown(wait=True)

    # Now, and only now, is this safe: the workers held duplicates of the tracker
    # pipe and are gone, so the tracker will actually exit. See
    # stop_resource_tracker for what it buys.
    stop_resource_tracker()


def _reverse_relu_one_region(
        region: ConstraintSet,
        W: np.ndarray,
        bias: np.ndarray,
        pruning: bool,
) -> tuple[dict[tuple, ConstraintSet], dict]:
    """
    One worker's whole job: reverse the ReLU over one region.

    At module scope, and taking the region as its first argument, so that
    functools.partial over it pickles for the process backends. A closure or a
    bound method would not.
    """
    if pruning:
        return region.reverse_relu_with_pruning(W, bias, return_metadata=True)
    return region.reverse_relu(W, bias, return_metadata=True)


def reverse_relu_layer(
        regions,
        W: np.ndarray,
        bias: np.ndarray,
        pruning: bool = True,
        executor=None,
        return_metadata: bool = False,
):
    """
    Reverse one ReLU layer over a whole list of regions, spreading the regions
    over workers.

    This is the parallel form of

        [region.reverse_relu_with_pruning(W, bias) for region in regions]

    and returns exactly that, in exactly that order. Every region's pruned search
    runs whole inside one worker, so it prunes precisely as it would have on its
    own and the answer is identical, not merely equivalent.

    @param regions:         the regions entering the layer, each a ConstraintSet
                            over the same variable
    @param W:               shape (big, small), passed through to reverse_relu
    @param bias:            shape (big,), passed through to reverse_relu
    @param pruning:         True (default) runs reverse_relu_with_pruning on each
                            region, False the exhaustive reverse_relu. The two
                            return the same regions; this is here so a timing run
                            can put the exhaustive search on the same workers
    @param executor:        None or 1 for a serial loop, an int for that many
                            joblib workers, or an Executor
    @param return_metadata: also return the aggregate of the per-region metadata,
                            see aggregate_metadata

    @return:                a list of the {zeroed_dim_idxs: ConstraintSet} dicts,
                            one per input region, or that list paired with the
                            aggregate metadata
    """
    executor = resolve_executor(executor)
    regions = list(regions)

    work = partial(_reverse_relu_one_region, W=W, bias=bias, pruning=pruning)

    start = time.time()
    results = executor.map(work, regions)
    wall_clock_seconds = time.time() - start

    region_dicts = [region_dict for region_dict, _ in results]
    if not return_metadata:
        return region_dicts

    metadata = aggregate_metadata(
        [metadata for _, metadata in results],
        wall_clock_seconds=wall_clock_seconds,
    )
    metadata['num_workers'] = executor.num_workers
    return region_dicts, metadata


def flatten_layer(region_dicts, substitute=None) -> list[ConstraintSet]:
    """
    Flatten what reverse_relu_layer returned into the list of regions the next
    layer back starts from.

    @param substitute:  applied to each pre-activation region on the way out,
                        which is where reverse_add_up_proj or reverse_up_proj
                        goes. Runs in this process, not in the workers: it is a
                        couple of matrix products against a region that has just
                        been shipped back, so sending it out again would cost more
                        than it saves. Omit it to get the pre-activation regions
                        themselves.
    """
    flattened = []
    for region_dict in region_dicts:
        for region in region_dict.values():
            flattened.append(region if substitute is None else substitute(region))
    return flattened
