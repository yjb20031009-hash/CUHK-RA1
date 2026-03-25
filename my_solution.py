"""Python/JAX orchestration port of MATLAB `my_solution.m`.

This module reproduces the script structure in a callable form:
- quick test call
- three optimization bundles (5-param, 7-param, 8-param)
- optional post-check evaluations
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cmaes2_jax import CMAESOptions, cmaes2_minimize
from .my_estimation_prepostdid1 import my_estimation_prepostdid1
from .my_estimation_prepostdid1_high import my_estimation_prepostdid1_high
from .my_estimation_prepostdid1_low import my_estimation_prepostdid1_low


Estimator = Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]]


@dataclass
class MySolutionResult:
    quick_test_value: float | None
    optimized: dict
    evaluations: dict


@dataclass
class FiveParamTestResult:
    optimized: dict
    evaluations: dict
    runtime_sec: dict


def _objective(estimator: Estimator):
    def f(x):
        ggvalue, _, _ = estimator(np.asarray(x, dtype=float))
        return ggvalue

    return f


def _run_one(estimator, x0: np.ndarray, sigma0: np.ndarray, opts: CMAESOptions, **kwargs):
    obj = lambda x: estimator(np.asarray(x, dtype=float), **kwargs)[0]
    t0 = time.time()
    res = cmaes2_minimize(obj, x0=np.asarray(x0, dtype=float), sigma0=np.asarray(sigma0, dtype=float), opts=opts)
    return {
        "x": res.xmin,
        "f": res.fmin,
        "iters": res.iterations,
        "evals": res.evaluations,
        "success": res.success,
        "time_sec": float(time.time() - t0),
    }


def _set_xla_gpu_autotune_level(level: int) -> None:
    """Set `--xla_gpu_autotune_level` in XLA_FLAGS for the current process."""
    key = "--xla_gpu_autotune_level="
    current = os.environ.get("XLA_FLAGS", "")
    toks = [t for t in current.split() if t and not t.startswith(key)]
    toks.append(f"{key}{int(level)}")
    os.environ["XLA_FLAGS"] = " ".join(toks)


def run_5param_test(
    *,
    opts: CMAESOptions | None = None,
    moments_full_path: str = "Sample_did_nosample.mat",
    moments_high_path: str = "Sample_did_nosample_high.mat",
    moments_low_path: str = "Sample_did_nosample_low.mat",
) -> FiveParamTestResult:
    """Run MATLAB-aligned 5-parameter tests for full/low/high DID variants."""
    opts = opts or CMAESOptions(max_iter=8, tol_x=1e-2, stop_fitness=1e-2, lbounds=0.01, ubounds=0.99)

    optimized: dict[str, dict] = {}
    evaluations: dict[str, float] = {}
    runtime_sec: dict[str, float] = {}

    optimized["did1_5param_full"] = _run_one(
        my_estimation_prepostdid1,
        np.array([0.39094, 0.5168, 0.30887, 0.56021, 0.81398]),
        np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        opts,
        moments_path=moments_full_path,
    )
    optimized["did1_5param_low"] = _run_one(
        my_estimation_prepostdid1_low,
        np.array([0.36106, 0.70974, 0.36162, 0.80249, 0.66785]),
        np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        opts,
        moments_path=moments_low_path,
    )
    optimized["did1_5param_high"] = _run_one(
        my_estimation_prepostdid1_high,
        np.array([0.10646, 0.24896, 0.12587, 0.75735, 0.61842]),
        np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        opts,
        moments_path=moments_high_path,
    )

    for name, item in optimized.items():
        gg, _, _ = (
            my_estimation_prepostdid1(item["x"], moments_path=moments_full_path)
            if "full" in name
            else (
                my_estimation_prepostdid1_low(item["x"], moments_path=moments_low_path)
                if "low" in name
                else my_estimation_prepostdid1_high(item["x"], moments_path=moments_high_path)
            )
        )
        evaluations[name] = float(gg)
        runtime_sec[name] = float(item["time_sec"])

    return FiveParamTestResult(optimized=optimized, evaluations=evaluations, runtime_sec=runtime_sec)


def run_my_solution(
    *,
    opts: CMAESOptions | None = None,
    moments_full_path: str = "Sample_did_nosample.mat",
    moments_high_path: str = "Sample_did_nosample_high.mat",
    moments_low_path: str = "Sample_did_nosample_low.mat",
    run_quick_test: bool = True,
    xla_gpu_autotune_level: int | None = None,
    run_5param: bool = True,
    run_7param: bool = True,
    run_8param: bool = False,
    quick_test_recompute_policy: bool = True,
) -> MySolutionResult:
    """Run optimization scenarios translated from MATLAB `my_solution.m`."""
    opts = opts or CMAESOptions(max_iter=8, tol_x=1e-2, stop_fitness=1e-2, lbounds=0.01, ubounds=0.99)
    if xla_gpu_autotune_level is not None:
        _set_xla_gpu_autotune_level(xla_gpu_autotune_level)

    optimized: dict[str, dict] = {}
    evaluations: dict[str, float] = {}
    quick_test_value: float | None = None

    if run_quick_test:
        quick_eval_param = np.array([0.2090, 0.11054, 0.6103, 0.9940, 0.9885, 0.3096, 0.3269, 0.2], dtype=float)
        quick_test_value, _, _ = my_estimation_prepostdid1_high(
            quick_eval_param,
            moments_path=moments_high_path,
            recompute_policy=quick_test_recompute_policy,
        )

    if run_5param:
        five_param_result = run_5param_test(
            opts=opts,
            moments_full_path=moments_full_path,
            moments_high_path=moments_high_path,
            moments_low_path=moments_low_path,
        )
        optimized.update(five_param_result.optimized)
        evaluations.update(five_param_result.evaluations)

    if run_7param:
        optimized["did1_7param_full"] = _run_one(
            my_estimation_prepostdid1,
            np.array([0.184507, 0.152546, 0.62454, 0.96237, 0.80, 0.24914, 0.53005]),
            np.array([0.1] * 7),
            opts,
            moments_path=moments_full_path,
        )
        optimized["did1_7param_low"] = _run_one(
            my_estimation_prepostdid1_low,
            np.array([0.25216, 0.18172, 0.73077, 0.95984, 0.84534, 0.16423, 0.52141]),
            np.array([0.1] * 7),
            opts,
            moments_path=moments_low_path,
        )
        optimized["did1_7param_high"] = _run_one(
            my_estimation_prepostdid1_high,
            np.array([0.156438, 0.112763, 0.71454, 0.98519, 0.69758, 0.34492, 0.55081]),
            np.array([0.1] * 7),
            opts,
            moments_path=moments_high_path,
        )

    if run_8param:
        x0 = np.array([0.20, 0.20, 0.60, 0.90, 0.70, 0.30, 0.40, 0.05])
        sig = np.array([0.08, 0.08, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06])
        optimized["did1_8param_full"] = _run_one(my_estimation_prepostdid1, x0, sig, opts, moments_path=moments_full_path)
        optimized["did1_8param_low"] = _run_one(my_estimation_prepostdid1_low, x0, sig, opts, moments_path=moments_low_path)
        optimized["did1_8param_high"] = _run_one(my_estimation_prepostdid1_high, x0, sig, opts, moments_path=moments_high_path)

    for name, item in optimized.items():
        if name in evaluations:
            continue
        gg, _, _ = (
            my_estimation_prepostdid1(item["x"], moments_path=moments_full_path)
            if "full" in name
            else (my_estimation_prepostdid1_low(item["x"], moments_path=moments_low_path) if "low" in name else my_estimation_prepostdid1_high(item["x"], moments_path=moments_high_path))
        )
        evaluations[name] = float(gg)

    return MySolutionResult(quick_test_value=quick_test_value, optimized=optimized, evaluations=evaluations)
