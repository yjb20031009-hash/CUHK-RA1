"""JAX rewrite of MATLAB `my_auxV_cal.m`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AuxVParams:
    t: int  # 0-based
    rho: float
    delta: float
    psi_1: float
    psi_2: float
    theta: float
    gyp: float
    adjcost: float
    ppt: float
    ppcost: float
    otcost: float
    income: float
    nn: int
    survprob: jnp.ndarray
    gret_sh: jnp.ndarray  # (nn, 3): [stock_ret, house_gross, weight]
    r: float
    model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    cash_min: float = 0.25
    cash_max: float = 19.9
    house_min: float = 0.25
    house_max: float = 19.9
    eq_atol: float = 1e-12


def my_auxv_cal(myinput: jnp.ndarray, p: AuxVParams, thecash: float, thehouse: float) -> jnp.ndarray:
    """Return scalar objective value, matching MATLAB `my_auxV_cal` logic."""
    myc, mya, myh = myinput[0], myinput[1], myinput[2]

    u = (1.0 - p.delta) * (myc**p.psi_1)

    house_gross = p.gret_sh[:, 1]
    housing_nn = myh * house_gross / p.gyp
    housing_nn = jnp.clip(housing_nn, p.house_min, p.house_max)

    # MATLAB uses exact compare; keep configurable tolerance for optimizer robustness.
    adjust_house = jnp.logical_not(jnp.isclose(myh, thehouse, atol=p.eq_atol, rtol=0.0))
    participate = mya > 0.0
    stock_ret = p.gret_sh[:, 0]

    def case_adj_part(_):
        sav = thecash + thehouse * (1.0 - p.adjcost - p.ppt) - myc - myh - p.ppcost - p.otcost
        return (sav * (1.0 - mya) * p.r + sav * mya * stock_ret) / p.gyp + p.income

    def case_adj_nopart(_):
        sav = thecash + thehouse * (1.0 - p.adjcost - p.ppt) - myc - myh
        cash = sav * p.r / p.gyp + p.income
        return jnp.full((p.nn,), cash)

    def case_noadj_part(_):
        sav = thecash + thehouse * (-p.ppt) - myc - p.ppcost - p.otcost
        return (sav * (1.0 - mya) * p.r + sav * mya * stock_ret) / p.gyp + p.income

    def case_noadj_nopart(_):
        sav = thecash + thehouse * (-p.ppt) - myc
        cash = sav * p.r / p.gyp + p.income
        return jnp.full((p.nn,), cash)

    cash_nn = jax.lax.cond(
        adjust_house,
        lambda _: jax.lax.cond(participate, case_adj_part, case_adj_nopart, operand=None),
        lambda _: jax.lax.cond(participate, case_noadj_part, case_noadj_nopart, operand=None),
        operand=None,
    )
    cash_nn = jnp.clip(cash_nn, p.cash_min, p.cash_max)

    int_v = p.model_fn(housing_nn, cash_nn)
    weights = p.gret_sh[:, 2]
    surv = p.survprob[p.t] if p.survprob.ndim == 1 else p.survprob[p.t, 0]

    aux_vv = (weights @ (int_v ** (1.0 - p.rho))) * surv
    aux_v = -((u + p.delta * (aux_vv ** (1.0 / p.theta))) ** p.psi_2)
    return aux_v
