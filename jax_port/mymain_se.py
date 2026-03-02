"""Approximate JAX/Python rewrite of MATLAB `mymain_se.m`.

This implementation keeps the original model flow but replaces MATLAB `fmincon`
with discrete grid search over candidate controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

from .my_auxv_cal import AuxVParams, my_auxv_cal
from .tauchen_hussey import tauchen_hussey


@dataclass(frozen=True)
class GridCfg:
    na: int = 5
    nc: int = 3
    nh2: int = 5
    n: int = 3
    ncash: int = 11
    nh: int = 6


@dataclass(frozen=True)
class LifeCfg:
    stept: int = 2
    tb: float = 10.0
    tr: float = 31.0
    td: float = 50.0


@dataclass(frozen=True)
class FixedParams:
    adjcost: float = 0.07
    maxhouse: float = 40.0
    minhouse: float = 0.0
    maxcash: float = 40.0
    mincash: float = 0.25
    minalpha: float = 0.01
    corr_hs: float = -0.08
    r: float = 1.05 - 0.048
    sigr: float = 0.42
    sigrh: float = 0.28
    incaa: float = 0.0
    incb1: float = 0.0
    incb2: float = 0.0
    incb3: float = 0.0
    ret_fac: float = 0.6
    minhouse2_value: float = 0.0


def _build_state_grids(fp: FixedParams, gcfg: GridCfg) -> tuple[jnp.ndarray, jnp.ndarray]:
    gcash = np.exp(np.linspace(np.log(fp.mincash), np.log(fp.maxcash), gcfg.ncash))
    ghouse = np.exp(np.linspace(np.log(fp.minhouse + 1.0), np.log(fp.maxhouse + 1.0), gcfg.nh)) - 1.0
    return jnp.asarray(gcash), jnp.asarray(ghouse)


def _minhouse2_normalized(fp: FixedParams) -> float:
    denom = np.exp(fp.incaa + fp.incb1 * 60 + fp.incb2 * 60**2 + fp.incb3 * 60**3) + np.exp(
        fp.incaa + fp.incb1 * 61 + fp.incb2 * 61**2 + fp.incb3 * 61**3
    )
    return float(fp.minhouse2_value / denom) if denom != 0 else 0.0


def _load_survprob(path_surv_mat: str) -> jnp.ndarray:
    d = loadmat(path_surv_mat)
    for k in ["survprob", "surv", "SurvProb"]:
        if k in d:
            return jnp.asarray(d[k])
    raise KeyError(f"Cannot find survprob in {path_surv_mat}. Keys={list(d.keys())}")


def _gret_sh(fp: FixedParams, grid: np.ndarray, weig: np.ndarray, mu: float, muh: float, n: int) -> jnp.ndarray:
    nn = n * n
    gret = fp.r + mu + grid * fp.sigr
    greth = np.zeros((n, n))
    for i1 in range(n):
        grid2 = grid[i1] * fp.corr_hs + grid * np.sqrt(1 - fp.corr_hs**2)
        greth[:, i1] = fp.r + muh + grid2 * fp.sigrh
    greth = greth.reshape(nn)

    out = np.zeros((nn, 3))
    for i in range(nn):
        out[i, 0] = gret[int(np.ceil((i + 1) / n)) - 1]
        out[i, 1] = greth[i]
        out[i, 2] = weig[int(np.ceil((i + 1) / n)) - 1] * weig[i % n]
    return jnp.asarray(out)


def _income_growth(fp: FixedParams, lcfg: LifeCfg, t: int) -> tuple[float, float]:
    if t + 1 >= int(lcfg.tr - lcfg.tb):
        return fp.ret_fac, 1.0
    age1 = lcfg.stept * (t + lcfg.tb + 1)
    age2 = lcfg.stept * (t + lcfg.tb)

    f_y1 = np.exp(fp.incaa + fp.incb1 * age1 + fp.incb2 * age1**2 + fp.incb3 * age1**3)
    f_y1_2 = np.exp(fp.incaa + fp.incb1 * (age1 + 1) + fp.incb2 * (age1 + 1) ** 2 + fp.incb3 * (age1 + 1) ** 3)
    f_y2 = np.exp(fp.incaa + fp.incb1 * age2 + fp.incb2 * age2**2 + fp.incb3 * age2**3)
    f_y2_2 = np.exp(fp.incaa + fp.incb1 * (age2 + 1) + fp.incb2 * (age2 + 1) ** 2 + fp.incb3 * (age2 + 1) ** 3)
    return 1.0, float(np.exp((f_y1 + f_y1_2) / (f_y2 + f_y2_2) - 1.0))


def _build_model_fn(v_next: jnp.ndarray, gcash: jnp.ndarray, ghouse: jnp.ndarray) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    from .interp2 import interp2_regular

    def model_fn(housing_nn: jnp.ndarray, cash_nn: jnp.ndarray) -> jnp.ndarray:
        return interp2_regular(ghouse, gcash, v_next.T, housing_nn, cash_nn, method="linear", bounds="clip")

    return model_fn


def _solve_one_state_discrete(thecash: float, thehouse: float, aux_params: AuxVParams, b: float, h_mode: str, can_participate: bool, gcfg: GridCfg, fp: FixedParams, minhouse2: float) -> tuple[float, float, float, float]:
    if b < 0.25:
        return (0.25, 0.0, 0.0, -jnp.inf)

    cgrid = jnp.linspace(0.25, max(b, 0.25), gcfg.nc)
    agrid = jnp.linspace(fp.minalpha, 1.0, gcfg.na) if can_participate else jnp.array([0.0])

    if h_mode == "keep":
        hgrid = jnp.array([thehouse])
        feas = lambda C, H: C <= b
    elif h_mode == "zero":
        hgrid = jnp.array([0.0])
        feas = lambda C, H: C + H <= b
    else:
        hgrid = jnp.linspace(minhouse2, fp.maxhouse, gcfg.nh2)
        feas = lambda C, H: C + H <= b

    C, A, H = jnp.meshgrid(cgrid, agrid, hgrid, indexing="ij")
    choices = jnp.stack([C, A, H], axis=-1).reshape(-1, 3)
    mask = feas(C, H).reshape(-1)

    vals = jax.vmap(lambda x: my_auxv_cal(x, aux_params, thecash, thehouse))(choices)
    vals = jnp.where(mask, vals, jnp.inf)
    idx = int(jnp.argmin(vals))
    best = choices[idx]
    return float(best[0]), float(best[1]), float(best[2]), float(-vals[idx])


def mymain_se(
    ppt: float,
    ppcost_in: float,
    otcost_in: float,
    rho: float,
    delta: float,
    psi: float,
    mu: float = 0.08,
    muh: float = 0.08,
    *,
    gcfg: GridCfg = GridCfg(),
    lcfg: LifeCfg = LifeCfg(),
    fp: FixedParams = FixedParams(),
    surv_mat_path: str = "surv.mat",
):
    tn = int(lcfg.td - lcfg.tb + 1)
    gcash, ghouse = _build_state_grids(fp, gcfg)
    survprob = _load_survprob(surv_mat_path)

    theta = (1.0 - rho) / (1.0 - 1.0 / psi)
    psi_1 = 1.0 - 1.0 / psi
    psi_2 = 1.0 / psi_1

    grid, weig2 = tauchen_hussey(gcfg.n, 0.0, 0.0, 1.0, 1.0)
    grid = np.asarray(grid).reshape(-1)
    weig = np.asarray(weig2)[0, :].reshape(-1)
    gret_sh = _gret_sh(fp, grid, weig, mu, muh, gcfg.n)

    C = np.zeros((gcfg.ncash, gcfg.nh, tn))
    A = np.ones((gcfg.ncash, gcfg.nh, tn))
    H = np.ones((gcfg.ncash, gcfg.nh, tn))
    V = np.zeros((gcfg.ncash, gcfg.nh, tn))
    C1, A1, H1, V1 = C.copy(), A.copy(), H.copy(), V.copy()

    for i in range(gcfg.ncash):
        for j in range(gcfg.nh):
            C[i, j, tn - 1] = float(gcash[i] + ghouse[j] * (1 - fp.adjcost - ppt))
            V[i, j, tn - 1] = C[i, j, tn - 1] * ((1.0 - delta) ** (psi / (psi - 1.0)))
    A[:, :, tn - 1] = 0.0
    H[:, :, tn - 1] = 0.0
    C1[:, :, tn - 1], A1[:, :, tn - 1], H1[:, :, tn - 1], V1[:, :, tn - 1] = (
        C[:, :, tn - 1],
        A[:, :, tn - 1],
        H[:, :, tn - 1],
        V[:, :, tn - 1],
    )

    def loop_block(V_next_np: np.ndarray, t: int, ppcost: float, otcost: float):
        income, gyp = _income_growth(fp, lcfg, t)
        ppcost *= gyp
        otcost *= gyp
        minhouse2 = _minhouse2_normalized(fp) * gyp

        aux_params = AuxVParams(
            t=t,
            rho=rho,
            delta=delta,
            psi_1=psi_1,
            psi_2=psi_2,
            theta=theta,
            gyp=gyp,
            adjcost=fp.adjcost,
            ppt=ppt,
            ppcost=ppcost,
            otcost=otcost,
            income=income,
            nn=gcfg.n * gcfg.n,
            survprob=survprob,
            gret_sh=gret_sh,
            r=fp.r,
            model_fn=_build_model_fn(jnp.asarray(V_next_np), gcash, ghouse),
        )
        return aux_params, ppcost, otcost, minhouse2

    ppcost = float(ppcost_in)
    otcost = 0.0
    for t in range(tn - 2, -1, -1):
        aux, ppcost, otcost, minhouse2 = loop_block(V[:, :, t + 1], t, ppcost, otcost)
        for i in range(gcfg.ncash):
            for j in range(gcfg.nh):
                cash, house = float(gcash[i]), float(ghouse[j])
                candidates = [
                    _solve_one_state_discrete(cash, house, aux, house * (1 - fp.adjcost - ppt) + cash - otcost - ppcost, "buy", True, gcfg, fp, minhouse2),
                    _solve_one_state_discrete(cash, house, aux, house * (1 - fp.adjcost - ppt) + cash - otcost - ppcost, "zero", True, gcfg, fp, minhouse2),
                    _solve_one_state_discrete(cash, house, aux, house * (1 - fp.adjcost - ppt) + cash, "buy", False, gcfg, fp, minhouse2),
                    _solve_one_state_discrete(cash, house, aux, house * (1 - fp.adjcost - ppt) + cash, "zero", False, gcfg, fp, minhouse2),
                    _solve_one_state_discrete(cash, house, aux, house * (-ppt) + cash - otcost - ppcost, "keep", True, gcfg, fp, minhouse2),
                    _solve_one_state_discrete(cash, house, aux, house * (-ppt) + cash, "keep", False, gcfg, fp, minhouse2),
                ]
                best = max(candidates, key=lambda z: z[3])
                C[i, j, t], A[i, j, t], H[i, j, t], V[i, j, t] = best

    ppcost = float(ppcost_in)
    otcost = float(otcost_in)
    for t in range(tn - 2, -1, -1):
        aux_pay, ppcost, otcost, minhouse2 = loop_block(V[:, :, t + 1], t, ppcost, otcost)
        aux_nopay = AuxVParams(**{**aux_pay.__dict__, "otcost": 0.0, "model_fn": _build_model_fn(jnp.asarray(V1[:, :, t + 1]), gcash, ghouse)})
        for i in range(gcfg.ncash):
            for j in range(gcfg.nh):
                cash, house = float(gcash[i]), float(ghouse[j])
                pay = max(
                    [
                        _solve_one_state_discrete(cash, house, aux_pay, house * (1 - fp.adjcost - ppt) + cash - otcost - ppcost, "buy", True, gcfg, fp, minhouse2),
                        _solve_one_state_discrete(cash, house, aux_pay, house * (1 - fp.adjcost - ppt) + cash - otcost - ppcost, "zero", True, gcfg, fp, minhouse2),
                        _solve_one_state_discrete(cash, house, aux_pay, house * (1 - fp.adjcost - ppt) + cash, "buy", False, gcfg, fp, minhouse2),
                        _solve_one_state_discrete(cash, house, aux_pay, house * (1 - fp.adjcost - ppt) + cash, "zero", False, gcfg, fp, minhouse2),
                        _solve_one_state_discrete(cash, house, aux_pay, house * (-ppt) + cash - otcost - ppcost, "keep", True, gcfg, fp, minhouse2),
                        _solve_one_state_discrete(cash, house, aux_pay, house * (-ppt) + cash, "keep", False, gcfg, fp, minhouse2),
                    ],
                    key=lambda z: z[3],
                )
                nopay = max(
                    [
                        _solve_one_state_discrete(cash, house, aux_nopay, house * (1 - fp.adjcost - ppt) + cash, "buy", False, gcfg, fp, minhouse2),
                        _solve_one_state_discrete(cash, house, aux_nopay, house * (1 - fp.adjcost - ppt) + cash, "zero", False, gcfg, fp, minhouse2),
                        _solve_one_state_discrete(cash, house, aux_nopay, house * (-ppt) + cash, "keep", False, gcfg, fp, minhouse2),
                    ],
                    key=lambda z: z[3],
                )
                best = pay if pay[3] >= nopay[3] else nopay
                C1[i, j, t], A1[i, j, t], H1[i, j, t], V1[i, j, t] = best

    H[H < 1e-3] = 0.0
    H1[H1 < 1e-3] = 0.0
    return C, A, H, C1, A1, H1
