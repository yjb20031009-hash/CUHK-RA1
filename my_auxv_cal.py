"""JAX rewrite of MATLAB `my_auxV_cal.m`."""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp

from interp2 import interp2_regular


def _f32(x):
    return jnp.asarray(x, dtype=jnp.float32)


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


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
    v_next: jnp.ndarray
    gcash_grid: jnp.ndarray
    ghouse_grid: jnp.ndarray
    cash_min: float = 0.25
    cash_max: float = 19.9
    house_min: float = 0.25
    house_max: float = 19.9
    eq_atol: float = 1e-12


@jax.jit
def _my_auxv_cal_jit(
    myinput: jnp.ndarray,
    thecash: float,
    thehouse: float,
    *,
    v_next: jnp.ndarray,
    gcash_grid: jnp.ndarray,
    ghouse_grid: jnp.ndarray,
    t: int,
    rho: float,
    delta: float,
    psi_1: float,
    psi_2: float,
    theta: float,
    gyp: float,
    adjcost: float,
    ppt: float,
    ppcost: float,
    otcost: float,
    income: float,
    survprob: jnp.ndarray,
    gret_sh: jnp.ndarray,
    r: float,
    cash_min: float,
    cash_max: float,
    house_min: float,
    house_max: float,
    eq_atol: float,
) -> jnp.ndarray:
    myc, mya, myh = myinput[0], myinput[1], myinput[2]

    u = (1.0 - delta) * (myc**psi_1)

    house_gross = gret_sh[:, 1]
    housing_nn = myh * house_gross / gyp
    housing_nn = jnp.clip(housing_nn, house_min, house_max)

    adjust_house = jnp.logical_not(jnp.isclose(myh, thehouse, atol=eq_atol, rtol=0.0))
    participate = mya > 0.0
    stock_ret = gret_sh[:, 0]

    def case_adj_part(_):
        sav = thecash + thehouse * (1.0 - adjcost - ppt) - myc - myh - ppcost - otcost
        return (sav * (1.0 - mya) * r + sav * mya * stock_ret) / gyp + income

    def case_adj_nopart(_):
        sav = thecash + thehouse * (1.0 - adjcost - ppt) - myc - myh
        cash = sav * r / gyp + income
        return jnp.ones_like(stock_ret) * cash

    def case_noadj_part(_):
        sav = thecash + thehouse * (-ppt) - myc - ppcost - otcost
        return (sav * (1.0 - mya) * r + sav * mya * stock_ret) / gyp + income

    def case_noadj_nopart(_):
        sav = thecash + thehouse * (-ppt) - myc
        cash = sav * r / gyp + income
        return jnp.ones_like(stock_ret) * cash

    cash_nn = jax.lax.cond(
        adjust_house,
        lambda _: jax.lax.cond(participate, case_adj_part, case_adj_nopart, operand=None),
        lambda _: jax.lax.cond(participate, case_noadj_part, case_noadj_nopart, operand=None),
        operand=None,
    )
    cash_nn = jnp.clip(cash_nn, cash_min, cash_max)

    int_v = interp2_regular(ghouse_grid, gcash_grid, v_next, housing_nn, cash_nn, method="linear", bounds="clip")
    weights = gret_sh[:, 2]
    surv = survprob[t] if survprob.ndim == 1 else survprob[t, 0]

    aux_vv = (weights @ (int_v ** (1.0 - rho))) * surv
    return -((u + delta * (aux_vv ** (1.0 / theta))) ** psi_2)


def my_auxv_cal(myinput: jnp.ndarray, p: AuxVParams, thecash: float, thehouse: float) -> jnp.ndarray:
    """Return scalar objective value, matching MATLAB `my_auxV_cal` logic."""
    return _my_auxv_cal_jit(
        _f32(myinput),
        _f32(thecash),
        _f32(thehouse),
        v_next=_f32(p.v_next),
        gcash_grid=_f32(p.gcash_grid),
        ghouse_grid=_f32(p.ghouse_grid),
        t=_i32(p.t),
        rho=_f32(p.rho),
        delta=_f32(p.delta),
        psi_1=_f32(p.psi_1),
        psi_2=_f32(p.psi_2),
        theta=_f32(p.theta),
        gyp=_f32(p.gyp),
        adjcost=_f32(p.adjcost),
        ppt=_f32(p.ppt),
        ppcost=_f32(p.ppcost),
        otcost=_f32(p.otcost),
        income=_f32(p.income),
        survprob=_f32(p.survprob),
        gret_sh=_f32(p.gret_sh),
        r=_f32(p.r),
        cash_min=_f32(p.cash_min),
        cash_max=_f32(p.cash_max),
        house_min=_f32(p.house_min),
        house_max=_f32(p.house_max),
        eq_atol=_f32(p.eq_atol),
    )
