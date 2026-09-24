"""Run one job per video/subject in FISH_WORKERS processes, with a progress bar.

Shared by every per-video / per-subject step (scene, track, ...), so they all
parallelise, fall back and report progress the same way.
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Sequence, TypeVar

import cv2
from tqdm import tqdm

log = logging.getLogger(__name__)

Job = TypeVar("Job")
Result = TypeVar("Result")


def _init_worker() -> None:
    """One OpenCV thread per worker process: the parallelism comes from the processes."""
    cv2.setNumThreads(1)


def run_parallel(func: Callable[[Job], Result], jobs: Sequence[Job], workers: int, desc: str) -> list[Result]:
    """Return ``[func(job) for job in jobs]``, computed in `workers` processes (here when 1).

    `func` must be a module-level function and jobs plain picklable values, so they can
    be sent to another process. The bar counts finished jobs; `disable=None` hides it
    when the output is not a terminal (logs, tests).
    """
    bar = tqdm(total=len(jobs), desc=desc, unit="job", disable=None)
    if workers <= 1 or len(jobs) <= 1:
        results = []
        for job in jobs:
            results.append(func(job))
            bar.update()
        bar.close()
        return results
    try:
        # spawn on every OS, not Linux's default fork: a forked child inherits OpenCV's
        # thread pool without its threads and hangs on its first cv2 call.
        pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=_init_worker,
                                   mp_context=multiprocessing.get_context("spawn"))
    except (OSError, NotImplementedError) as error:
        # Some locked-down environments forbid the semaphores a process pool needs.
        log.warning("cannot start %d worker processes (%s); running jobs one by one", workers, error)
        bar.close()
        return run_parallel(func, jobs, 1, desc)
    results: dict[int, Result] = {}
    with pool, bar:
        futures = {pool.submit(func, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            bar.update()
    return [results[i] for i in range(len(jobs))]  # back in job order
