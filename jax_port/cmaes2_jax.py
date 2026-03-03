"""Engineering-first JAX port of core CMA-ES logic from MATLAB `cmaes2.m`.

This is a compact, practical CMA-ES implementation that keeps the usual API spirit:
- minimize(fun, x0, sigma0, opts)
- bound handling via clipping
- rank-mu covariance update + CSA step-size control
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray


@dataclass
class CMAESOptions:
    max_iter: int = 200
    tol_x: float = 1e-8
    stop_fitness: float | None = None
    seed: int = 0
    popsize: int | None = None
    lbounds: float | Array | None = None
    ubounds: float | Array | None = None


@dataclass
class CMAESResult:
    xmin: np.ndarray
    fmin: float
    iterations: int
    evaluations: int
    success: bool


def _as_bounds(val, n, default):
    if val is None:
        return jnp.full((n,), default)
    arr = jnp.asarray(val)
    if arr.ndim == 0:
        return jnp.full((n,), float(arr))
    return arr


def cmaes2_minimize(
    fun: Callable[[Array], float | Array],
    x0,
    sigma0,
    opts: CMAESOptions | None = None,
) -> CMAESResult:
    """Minimize objective with CMA-ES (JAX-based sampling, NumPy sorting)."""
    opts = opts or CMAESOptions()
    xmean = jnp.asarray(x0, dtype=jnp.float64)
    n = int(xmean.shape[0])

    sigma = float(np.asarray(sigma0).reshape(-1)[0]) if np.asarray(sigma0).size == 1 else float(np.mean(np.asarray(sigma0)))

    lam = opts.popsize or (4 + int(3 * np.log(n)))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = float(np.sum(weights) ** 2 / np.sum(weights**2))

    # Strategy parameters (standard CMA-ES)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    pc = jnp.zeros((n,), dtype=jnp.float64)
    ps = jnp.zeros((n,), dtype=jnp.float64)
    C = jnp.eye(n, dtype=jnp.float64)
    B = jnp.eye(n, dtype=jnp.float64)
    D = jnp.ones((n,), dtype=jnp.float64)
    invsqrtC = jnp.eye(n, dtype=jnp.float64)
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    lb = _as_bounds(opts.lbounds, n, -jnp.inf)
    ub = _as_bounds(opts.ubounds, n, jnp.inf)

    key = jax.random.key(opts.seed)
    counteval = 0
    best_x = np.asarray(xmean)
    best_f = np.inf

    for it in range(1, opts.max_iter + 1):
        key, sub = jax.random.split(key)
        arz = jax.random.normal(sub, (lam, n), dtype=jnp.float64)
        ary = arz @ (B * D).T
        arx = xmean[None, :] + sigma * ary
        arx = jnp.clip(arx, lb, ub)

        # Objective eval (python loop for compatibility with arbitrary fun)
        fit = np.array([float(np.asarray(fun(arx[k]))) for k in range(lam)], dtype=float)
        counteval += lam

        idx = np.argsort(fit)
        xold = xmean
        xsel = arx[idx[:mu]]
        zsel = arz[idx[:mu]]
        xmean = jnp.asarray(np.sum(np.asarray(weights)[:, None] * np.asarray(xsel), axis=0))
        zmean = jnp.asarray(np.sum(np.asarray(weights)[:, None] * np.asarray(zsel), axis=0))

        # Evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ zmean)
        hsig = float((jnp.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * it))) / chi_n < (1.4 + 2 / (n + 1)))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)

        # Covariance update
        artmp = (xsel - xold[None, :]) / sigma
        C = (1 - c1 - cmu) * C + c1 * (jnp.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
        for i in range(mu):
            C = C + cmu * weights[i] * jnp.outer(artmp[i], artmp[i])

        sigma = sigma * np.exp((cs / damps) * (float(jnp.linalg.norm(ps)) / chi_n - 1))

        # Decomposition refresh each iter (simple/robust)
        eigvals, eigvecs = np.linalg.eigh(np.asarray(C))
        eigvals = np.maximum(eigvals, 1e-30)
        D = jnp.asarray(np.sqrt(eigvals))
        B = jnp.asarray(eigvecs)
        invsqrtC = jnp.asarray(eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T)

        if fit[idx[0]] < best_f:
            best_f = float(fit[idx[0]])
            best_x = np.asarray(arx[idx[0]])

        if opts.stop_fitness is not None and best_f <= opts.stop_fitness:
            return CMAESResult(best_x, best_f, it, counteval, True)
        if sigma * float(np.max(np.asarray(D))) < opts.tol_x:
            return CMAESResult(best_x, best_f, it, counteval, True)

    return CMAESResult(best_x, best_f, opts.max_iter, counteval, False)
