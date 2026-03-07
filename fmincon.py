"""MATLAB-like fmincon wrapper using SciPy + JAX gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, minimize


@dataclass
class FminconResult:
    x: np.ndarray
    fval: float
    exitflag: int
    output: dict
    lambda_: dict


def _wrap_fun(fun: Callable[[jnp.ndarray], jnp.ndarray | float]) -> Callable[[np.ndarray], float]:
    def wrapped(x: np.ndarray) -> float:
        val = fun(jnp.asarray(x))
        return float(np.asarray(val))

    return wrapped


def _wrap_jac(fun):
    grad_fun = jax.grad(lambda z: jnp.asarray(fun(z), dtype=jnp.float64))

    def wrapped(x: np.ndarray) -> np.ndarray:
        return np.asarray(grad_fun(jnp.asarray(x)), dtype=float)

    return wrapped


def fmincon(
    fun: Callable[[jnp.ndarray], jnp.ndarray | float],
    x0: Sequence[float],
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
    Aeq: np.ndarray | None = None,
    beq: np.ndarray | None = None,
    lb: Sequence[float] | None = None,
    ub: Sequence[float] | None = None,
    nonlcon: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None = None,
    options: dict | None = None,
) -> FminconResult:
    """Approximate MATLAB fmincon API.

    nonlcon should return (c, ceq) where c<=0 and ceq==0.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    options = options or {}

    method = options.get("Algorithm", "sqp").lower()
    scipy_method = "SLSQP" if method in {"sqp", "active-set", "interior-point"} else "trust-constr"

    constraints = []

    if A is not None and b is not None:
        constraints.append(LinearConstraint(np.asarray(A, dtype=float), -np.inf, np.asarray(b, dtype=float)))
    if Aeq is not None and beq is not None:
        beq_arr = np.asarray(beq, dtype=float)
        constraints.append(LinearConstraint(np.asarray(Aeq, dtype=float), beq_arr, beq_arr))

    if nonlcon is not None:
        def c_fun(x):
            c, _ = nonlcon(jnp.asarray(x))
            return np.asarray(c, dtype=float)

        def ceq_fun(x):
            _, ceq = nonlcon(jnp.asarray(x))
            return np.asarray(ceq, dtype=float)

        constraints.append(NonlinearConstraint(c_fun, -np.inf, 0.0))
        constraints.append(NonlinearConstraint(ceq_fun, 0.0, 0.0))

    if lb is None:
        lb = -np.inf * np.ones(n)
    if ub is None:
        ub = np.inf * np.ones(n)
    bounds = Bounds(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))

    res: OptimizeResult = minimize(
        _wrap_fun(fun),
        x0,
        method=scipy_method,
        jac=_wrap_jac(fun),
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": int(options.get("MaxIterations", 1000)),
            "ftol": float(options.get("OptimalityTolerance", 1e-8)),
            "disp": bool(options.get("Display", False)),
        },
    )

    exitflag = 1 if res.success else 0
    output = {
        "iterations": getattr(res, "nit", None),
        "funcCount": getattr(res, "nfev", None),
        "message": str(res.message),
    }

    lam = {
        "ineqlin": None,
        "eqlin": None,
        "ineqnonlin": None,
        "eqnonlin": None,
        "lower": None,
        "upper": None,
    }

    return FminconResult(x=np.asarray(res.x), fval=float(res.fun), exitflag=exitflag, output=output, lambda_=lam)
