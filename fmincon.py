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
    # Pre-compile objective gradient once to reduce Python/JAX dispatch overhead
    grad_fun = jax.jit(jax.grad(lambda z: jnp.asarray(fun(z), dtype=jnp.float64)))

    def wrapped(x: np.ndarray) -> np.ndarray:
        return np.asarray(grad_fun(jnp.asarray(x)), dtype=float)

    return wrapped


def _constraint_violation(
    x: np.ndarray,
    A: np.ndarray | None,
    b: np.ndarray | None,
    Aeq: np.ndarray | None,
    beq: np.ndarray | None,
    nonlcon: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None,
) -> float:
    """Return max absolute constraint violation (0 means feasible)."""
    v = 0.0
    xx = np.asarray(x, dtype=float)
    if A is not None and b is not None:
        v = max(v, float(np.max(np.maximum(np.asarray(A, dtype=float) @ xx - np.asarray(b, dtype=float), 0.0))))
    if Aeq is not None and beq is not None:
        v = max(v, float(np.max(np.abs(np.asarray(Aeq, dtype=float) @ xx - np.asarray(beq, dtype=float)))))
    if nonlcon is not None:
        c, ceq = nonlcon(jnp.asarray(xx))
        c_np = np.asarray(c, dtype=float)
        ceq_np = np.asarray(ceq, dtype=float)
        if c_np.size:
            v = max(v, float(np.max(np.maximum(c_np, 0.0))))
        if ceq_np.size:
            v = max(v, float(np.max(np.abs(ceq_np))))
    return float(v)



def _build_slsqp_constraints(
    A: np.ndarray | None,
    b: np.ndarray | None,
    Aeq: np.ndarray | None,
    beq: np.ndarray | None,
    nonlcon: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None,
):
    """Build old-style SciPy SLSQP constraints dicts (ineq: g(x)>=0, eq: h(x)=0)."""
    constraints: list[dict] = []

    if A is not None and b is not None:
        A_arr = np.atleast_2d(np.asarray(A, dtype=float))
        b_arr = np.asarray(b, dtype=float).reshape(-1)

        def lin_ineq_fun(x):
            return b_arr - A_arr @ np.asarray(x, dtype=float)

        def lin_ineq_jac(_x):
            return -A_arr

        constraints.append({"type": "ineq", "fun": lin_ineq_fun, "jac": lin_ineq_jac})

    if Aeq is not None and beq is not None:
        Aeq_arr = np.atleast_2d(np.asarray(Aeq, dtype=float))
        beq_arr = np.asarray(beq, dtype=float).reshape(-1)

        def lin_eq_fun(x):
            return Aeq_arr @ np.asarray(x, dtype=float) - beq_arr

        def lin_eq_jac(_x):
            return Aeq_arr

        constraints.append({"type": "eq", "fun": lin_eq_fun, "jac": lin_eq_jac})

    if nonlcon is not None:
        c_jac_jit = jax.jit(jax.jacobian(lambda z: jnp.asarray(nonlcon(z)[0], dtype=jnp.float64)))
        ceq_jac_jit = jax.jit(jax.jacobian(lambda z: jnp.asarray(nonlcon(z)[1], dtype=jnp.float64)))

        def c_fun(x):
            c, _ = nonlcon(jnp.asarray(x))
            return -np.asarray(c, dtype=float)

        def c_jac(x):
            return -np.asarray(c_jac_jit(jnp.asarray(x)), dtype=float)

        def ceq_fun(x):
            _, ceq = nonlcon(jnp.asarray(x))
            return np.asarray(ceq, dtype=float)

        def ceq_jac(x):
            return np.asarray(ceq_jac_jit(jnp.asarray(x)), dtype=float)

        constraints.append({"type": "ineq", "fun": c_fun, "jac": c_jac})
        constraints.append({"type": "eq", "fun": ceq_fun, "jac": ceq_jac})

    return constraints


def _build_trust_constraints(
    A: np.ndarray | None,
    b: np.ndarray | None,
    Aeq: np.ndarray | None,
    beq: np.ndarray | None,
    nonlcon: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None,
):
    """Build new-style trust-constr constraints."""
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

        c_jac_jit = jax.jit(jax.jacobian(lambda z: jnp.asarray(nonlcon(z)[0], dtype=jnp.float64)))
        ceq_jac_jit = jax.jit(jax.jacobian(lambda z: jnp.asarray(nonlcon(z)[1], dtype=jnp.float64)))

        def c_jac(x):
            return np.asarray(c_jac_jit(jnp.asarray(x)), dtype=float)

        def ceq_jac(x):
            return np.asarray(ceq_jac_jit(jnp.asarray(x)), dtype=float)

        constraints.append(NonlinearConstraint(c_fun, -np.inf, 0.0, jac=c_jac))
        constraints.append(NonlinearConstraint(ceq_fun, 0.0, 0.0, jac=ceq_jac))
    return constraints

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
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
) -> FminconResult:
    """Approximate MATLAB fmincon API.

    nonlcon should return (c, ceq) where c<=0 and ceq==0.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    options = options or {}

    method = options.get("Algorithm", "sqp").lower()
    scipy_method = "SLSQP" if method in {"sqp", "active-set", "interior-point"} else "trust-constr"

    if scipy_method == "SLSQP":
        constraints = _build_slsqp_constraints(A, b, Aeq, beq, nonlcon)
    else:
        constraints = _build_trust_constraints(A, b, Aeq, beq, nonlcon)

    if lb is None:
        lb = -np.inf * np.ones(n)
    if ub is None:
        ub = np.inf * np.ones(n)
    bounds = Bounds(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))

    objective_jac = jac if jac is not None else _wrap_jac(fun)

    optimality_tol = float(options.get("OptimalityTolerance", options.get("TolFun", 1e-8)))
    constraint_tol = options.get("ConstraintTolerance", options.get("TolCon", None))
    constraint_tol = float(constraint_tol) if constraint_tol is not None else None
    # SLSQP only exposes a single ftol; approximate MATLAB split tolerances by
    # using the tighter of optimality/constraint tolerances when both are provided.
    scipy_ftol = min(optimality_tol, constraint_tol) if constraint_tol is not None else optimality_tol

    res: OptimizeResult = minimize(
        _wrap_fun(fun),
        x0,
        method=scipy_method,
        jac=objective_jac,
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": int(options.get("MaxIterations", options.get("MaxIter", 1000))),
            "ftol": float(scipy_ftol),
            "disp": bool(options.get("Display", False)),
        },
    )

    maxcv = _constraint_violation(np.asarray(res.x, dtype=float), A, b, Aeq, beq, nonlcon)
    cv_ok = True if constraint_tol is None else (maxcv <= constraint_tol)
    exitflag = 1 if (res.success and cv_ok) else 0
    output = {
        "iterations": getattr(res, "nit", None),
        "funcCount": getattr(res, "nfev", None),
        "message": str(res.message),
        "constrviolation": maxcv,
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
