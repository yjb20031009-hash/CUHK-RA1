"""Approximate JAX/Python rewrite of MATLAB `mymain_se.m`.

阅读指南：
1) 先看 `mymain_se` 主函数（终值 -> 两个 backward loop）。
2) 再看 `_solve_one_state_discrete`（单个状态点如何挑最优 choice）。
3) 最后看 `AuxVParams + my_auxv_cal`（给定 choice 如何算目标值）。

注意：这里用“离散候选网格搜索”替代了 MATLAB `fmincon` 连续优化，
因此是近似求解而非完全数值等价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

from .my_auxv_cal import AuxVParams, my_auxv_cal
from .tauchen_hussey import tauchen_hussey


@dataclass(frozen=True)
class GridCfg:
    """离散搜索与状态空间网格配置。"""

    na: int = 5  # 股票占比 alpha 候选点数量
    nc: int = 3  # 消费 c 候选点数量
    nh2: int = 5  # 调整后房产 h 候选点数量
    n: int = 3  # Tauchen-Hussey 冲击离散点数量（单维）
    ncash: int = 11  # cash 状态网格数量
    nh: int = 6  # house 状态网格数量


@dataclass(frozen=True)
class LifeCfg:
    """生命周期时间参数（与 MATLAB 默认一致）。"""

    stept: int = 2
    tb: float = 10.0
    tr: float = 31.0
    td: float = 50.0


@dataclass(frozen=True)
class FixedParams:
    """模型固定参数（非估计参数）。"""

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
    """构造 cash / house 的状态网格（log spacing 对应 MATLAB 写法）。"""
    gcash = np.exp(np.linspace(np.log(fp.mincash), np.log(fp.maxcash), gcfg.ncash))
    ghouse = np.exp(np.linspace(np.log(fp.minhouse + 1.0), np.log(fp.maxhouse + 1.0), gcfg.nh)) - 1.0
    return jnp.asarray(gcash), jnp.asarray(ghouse)


def _minhouse2_normalized(fp: FixedParams) -> float:
    """最小购房门槛标准化（与 MATLAB 公式保持一致）。"""
    denom = np.exp(fp.incaa + fp.incb1 * 60 + fp.incb2 * 60**2 + fp.incb3 * 60**3) + np.exp(
        fp.incaa + fp.incb1 * 61 + fp.incb2 * 61**2 + fp.incb3 * 61**3
    )
    return float(fp.minhouse2_value / denom) if denom != 0 else 0.0


def _load_survprob(path_surv_mat: str) -> jnp.ndarray:
    """从 mat 文件读取生存概率矩阵。"""
    d = loadmat(path_surv_mat)
    for k in ["survprob", "surv", "SurvProb"]:
        if k in d:
            return jnp.asarray(d[k])
    raise KeyError(f"Cannot find survprob in {path_surv_mat}. Keys={list(d.keys())}")


def _gret_sh(fp: FixedParams, grid: np.ndarray, weig: np.ndarray, mu: float, muh: float, n: int) -> jnp.ndarray:
    """构造联合收益冲击矩阵 gret_sh: [stock_ret, house_ret, prob_weight]。"""
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
    """给定期 t，返回当期 income 与收入增长因子 gyp。"""
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
    """把下一期价值函数数组封装成可调用 model_fn(housing, cash)。"""
    from .interp2 import interp2_regular

    def model_fn(housing_nn: jnp.ndarray, cash_nn: jnp.ndarray) -> jnp.ndarray:
        # 注意插值轴：x=ghouse, y=gcash, V shape=(len(y), len(x))
        # v_next 在本实现中已是 (ncash, nh) = (len(gcash), len(ghouse))，不应再转置。
        return interp2_regular(ghouse, gcash, v_next, housing_nn, cash_nn, method="linear", bounds="clip")

    return model_fn


def _build_model_fn_spline(v_next_np: np.ndarray, gcash_np: np.ndarray, ghouse_np: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a MATLAB-like spline interpolator: value(cash, house) -> scalar/array."""
    # MATLAB: griddedInterpolant({gcash, ghouse}, V_next, 'spline')
    kx = int(min(3, max(1, len(gcash_np) - 1)))
    ky = int(min(3, max(1, len(ghouse_np) - 1)))
    sp = RectBivariateSpline(gcash_np, ghouse_np, v_next_np, kx=kx, ky=ky, s=0.0)

    def model_fn(housing_nn: np.ndarray, cash_nn: np.ndarray) -> np.ndarray:
        h = np.asarray(housing_nn, dtype=float)
        c = np.asarray(cash_nn, dtype=float)
        c, h = np.broadcast_arrays(c, h)
        out = sp.ev(c.ravel(), h.ravel())
        return out.reshape(c.shape)

    return model_fn


def _my_auxv_cal_np(
    myinput: np.ndarray,
    p: AuxVParams,
    thecash: float,
    thehouse: float,
    model_fn_np: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    """NumPy/Scipy version of objective used by continuous optimizer."""
    myc, mya, myh = float(myinput[0]), float(myinput[1]), float(myinput[2])
    u = (1.0 - p.delta) * (myc ** p.psi_1)

    gret_sh = np.asarray(p.gret_sh)
    house_gross = gret_sh[:, 1]
    stock_ret = gret_sh[:, 0]
    weights = gret_sh[:, 2]

    housing_nn = np.clip(myh * house_gross / p.gyp, p.house_min, p.house_max)
    adjust_house = not np.isclose(myh, thehouse, atol=p.eq_atol, rtol=0.0)
    participate = mya > 0.0

    if adjust_house and participate:
        sav = thecash + thehouse * (1.0 - p.adjcost - p.ppt) - myc - myh - p.ppcost - p.otcost
        cash_nn = (sav * (1.0 - mya) * p.r + sav * mya * stock_ret) / p.gyp + p.income
    elif adjust_house and (not participate):
        sav = thecash + thehouse * (1.0 - p.adjcost - p.ppt) - myc - myh
        cash_nn = np.full_like(stock_ret, sav * p.r / p.gyp + p.income)
    elif (not adjust_house) and participate:
        sav = thecash + thehouse * (-p.ppt) - myc - p.ppcost - p.otcost
        cash_nn = (sav * (1.0 - mya) * p.r + sav * mya * stock_ret) / p.gyp + p.income
    else:
        sav = thecash + thehouse * (-p.ppt) - myc
        cash_nn = np.full_like(stock_ret, sav * p.r / p.gyp + p.income)

    cash_nn = np.clip(cash_nn, p.cash_min, p.cash_max)
    int_v = model_fn_np(housing_nn, cash_nn)
    surv = float(np.asarray(p.survprob)[p.t] if np.asarray(p.survprob).ndim == 1 else np.asarray(p.survprob)[p.t, 0])
    aux_vv = float(weights @ (int_v ** (1.0 - p.rho))) * surv
    return -((u + p.delta * (aux_vv ** (1.0 / p.theta))) ** p.psi_2)


def _solve_one_state_continuous(
    thecash: float,
    thehouse: float,
    aux_params: AuxVParams,
    b: float,
    h_mode: str,
    can_participate: bool,
    fp: FixedParams,
    minhouse2: float,
    model_fn_np: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[float, float, float, float]:
    """Continuous per-state solve (closer to MATLAB fmincon) via SLSQP."""
    if b < 0.25:
        return 0.25, 0.0, 0.0, -np.inf

    if h_mode == "keep":
        h_lb, h_ub = float(thehouse), float(thehouse)
    elif h_mode == "zero":
        h_lb, h_ub = 0.0, 0.0
    else:
        h_lb, h_ub = float(minhouse2), float(fp.maxhouse)

    if can_participate:
        a_lb, a_ub = float(fp.minalpha), 1.0
        a0 = max(a_lb, 0.2)
    else:
        a_lb, a_ub = 0.0, 0.0
        a0 = 0.0

    c_lb, c_ub = 0.25, max(float(b), 0.25)
    h0 = h_lb if h_lb == h_ub else min(max(h_lb, minhouse2 if h_mode == "buy" else 0.0), h_ub)
    if h_mode in {"buy", "zero"}:
        c0 = min(max(c_lb, 0.5 * max(b - h0, c_lb)), c_ub)
    else:
        c0 = min(max(c_lb, 0.5 * max(b, c_lb)), c_ub)
    x0 = np.array([c0, a0, h0], dtype=float)

    def obj(x):
        return float(_my_auxv_cal_np(np.asarray(x, dtype=float), aux_params, float(thecash), float(thehouse), model_fn_np))

    constraints = []
    if h_mode in {"buy", "zero"}:
        constraints.append({"type": "ineq", "fun": lambda x: float(b) - x[0] - x[2]})
    else:
        constraints.append({"type": "ineq", "fun": lambda x: float(b) - x[0]})

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=[(c_lb, c_ub), (a_lb, a_ub), (h_lb, h_ub)],
        constraints=constraints,
        options={"maxiter": 120, "ftol": 1e-7, "disp": False},
    )

    if not res.success:
        fval = obj(x0)
        return float(x0[0]), float(x0[1]), float(x0[2]), float(-fval)

    x = np.asarray(res.x, dtype=float)
    return float(x[0]), float(x[1]), float(x[2]), float(-res.fun)


def _solve_one_state_discrete(
    thecash: float,
    thehouse: float,
    aux_params: AuxVParams,
    b: float,
    h_mode: str,
    can_participate: bool,
    gcfg: GridCfg,
    fp: FixedParams,
    minhouse2: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """单状态点求解（JAX 友好版本，可被 vmap 批处理）。"""
    b = jnp.asarray(b)
    cgrid = jnp.linspace(0.25, jnp.maximum(b, 0.25), gcfg.nc)
    agrid = jnp.linspace(fp.minalpha, 1.0, gcfg.na) if can_participate else jnp.array([0.0])

    if h_mode == "keep":
        hgrid = jnp.array([thehouse])
        feas = lambda C, H: C <= b
    elif h_mode == "zero":
        hgrid = jnp.array([0.0])
        feas = lambda C, H: C + H <= b
    else:  # "buy": 在 [minhouse2, maxhouse] 上选房产
        hgrid = jnp.linspace(minhouse2, fp.maxhouse, gcfg.nh2)
        feas = lambda C, H: C + H <= b

    C, A, H = jnp.meshgrid(cgrid, agrid, hgrid, indexing="ij")
    choices = jnp.stack([C, A, H], axis=-1).reshape(-1, 3)
    mask = feas(C, H).reshape(-1) & (b >= 0.25)

    vals = jax.vmap(lambda x: my_auxv_cal(x, aux_params, thecash, thehouse))(choices)
    vals = jnp.where(mask, vals, jnp.inf)
    idx = jnp.argmin(vals)
    best = choices[idx]
    out = jnp.concatenate([best, jnp.array([-vals[idx]])])
    fallback = jnp.array([0.25, 0.0, 0.0, -jnp.inf])
    out = jnp.where(jnp.any(mask), out, fallback)
    return out[0], out[1], out[2], out[3]


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
    solver_mode: str = "continuous",
    surv_mat_path: str = "surv.mat",
):
    """主求解函数，对应 MATLAB `mymain_se`。

    返回：C, A, H, C1, A1, H1，形状均为 (ncash, nh, tn)

    结构：
    - 终值条件（t=最后一期）
    - Loop 1: 已支付 one-time cost 的人群（输出 C/A/H/V）
    - Loop 2: 未支付人群，对比“当期支付 vs 不支付”（输出 C1/A1/H1/V1）
    """
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

    # 终值条件：最后一期把可变现资源用于消费
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
        """给 backward 某一期准备 AuxVParams。"""
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
        model_fn_np = _build_model_fn_spline(np.asarray(V_next_np, dtype=float), np.asarray(gcash, dtype=float), np.asarray(ghouse, dtype=float))
        return aux_params, model_fn_np, ppcost, otcost, minhouse2

    cash_mesh, house_mesh = jnp.meshgrid(gcash, ghouse, indexing="ij")
    cash_flat = cash_mesh.reshape(-1)
    house_flat = house_mesh.reshape(-1)

    def _solve_case_batch(aux_params: AuxVParams, ppc: float, otc: float, minh2: float, *, h_mode: str, can_participate: bool, budget_fn):
        def _one(c, h):
            b = budget_fn(c, h, ppc, otc)
            return jnp.stack(_solve_one_state_discrete(c, h, aux_params, b, h_mode, can_participate, gcfg, fp, minh2))

        return jax.vmap(_one)(cash_flat, house_flat)  # (n_state, 4)

    # Loop 1: 已经支付过 one-time cost 的人群（批量化 state 维度）
    ppcost = float(ppcost_in)
    otcost = 0.0
    for t in range(tn - 2, -1, -1):
        aux, model_fn_np, ppcost, otcost, minhouse2 = loop_block(V[:, :, t + 1], t, ppcost, otcost)

        if solver_mode == "continuous":
            best_np = np.zeros((gcfg.ncash, gcfg.nh, 4), dtype=float)
            for i in range(gcfg.ncash):
                for j in range(gcfg.nh):
                    c, h = float(gcash[i]), float(ghouse[j])
                    cand = []
                    cand.append(_solve_one_state_continuous(c, h, aux, h * (1 - fp.adjcost - ppt) + c - otcost - ppcost, "buy", True, fp, minhouse2, model_fn_np))
                    cand.append(_solve_one_state_continuous(c, h, aux, h * (1 - fp.adjcost - ppt) + c - otcost - ppcost, "zero", True, fp, minhouse2, model_fn_np))
                    cand.append(_solve_one_state_continuous(c, h, aux, h * (1 - fp.adjcost - ppt) + c, "buy", False, fp, minhouse2, model_fn_np))
                    cand.append(_solve_one_state_continuous(c, h, aux, h * (1 - fp.adjcost - ppt) + c, "zero", False, fp, minhouse2, model_fn_np))
                    cand.append(_solve_one_state_continuous(c, h, aux, h * (-ppt) + c - otcost - ppcost, "keep", True, fp, minhouse2, model_fn_np))
                    cand.append(_solve_one_state_continuous(c, h, aux, h * (-ppt) + c, "keep", False, fp, minhouse2, model_fn_np))
                    best_np[i, j, :] = np.asarray(max(cand, key=lambda z: z[3]), dtype=float)
        else:
            case_stack = jnp.stack(
                [
                    _solve_case_batch(aux, ppcost, otcost, minhouse2, h_mode="buy", can_participate=True, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c - otc - ppc),
                    _solve_case_batch(aux, ppcost, otcost, minhouse2, h_mode="zero", can_participate=True, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c - otc - ppc),
                    _solve_case_batch(aux, ppcost, otcost, minhouse2, h_mode="buy", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c),
                    _solve_case_batch(aux, ppcost, otcost, minhouse2, h_mode="zero", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c),
                    _solve_case_batch(aux, ppcost, otcost, minhouse2, h_mode="keep", can_participate=True, budget_fn=lambda c, h, ppc, otc: h * (-ppt) + c - otc - ppc),
                    _solve_case_batch(aux, ppcost, otcost, minhouse2, h_mode="keep", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (-ppt) + c),
                ],
                axis=0,
            )
            best_idx = jnp.argmax(case_stack[:, :, 3], axis=0)
            best = jnp.take_along_axis(case_stack, best_idx[None, :, None], axis=0)[0]
            best_np = np.asarray(best).reshape(gcfg.ncash, gcfg.nh, 4)
        C[:, :, t], A[:, :, t], H[:, :, t], V[:, :, t] = best_np[:, :, 0], best_np[:, :, 1], best_np[:, :, 2], best_np[:, :, 3]

    # Loop 2: 未支付过 one-time cost 的人群（批量化 state 维度）
    # 对比“当期支付 otcost”与“继续不支付”两条路径，选价值更高者
    ppcost = float(ppcost_in)
    otcost = float(otcost_in)
    for t in range(tn - 2, -1, -1):
        aux_pay, model_pay_np, ppcost, otcost, minhouse2 = loop_block(V[:, :, t + 1], t, ppcost, otcost)
        aux_nopay = AuxVParams(
            **{
                **aux_pay.__dict__,
                "otcost": 0.0,
                "model_fn": _build_model_fn(jnp.asarray(V1[:, :, t + 1]), gcash, ghouse),
            }
        )
        model_nopay_np = _build_model_fn_spline(np.asarray(V1[:, :, t + 1], dtype=float), np.asarray(gcash, dtype=float), np.asarray(ghouse, dtype=float))

        if solver_mode == "continuous":
            best_np = np.zeros((gcfg.ncash, gcfg.nh, 4), dtype=float)
            for i in range(gcfg.ncash):
                for j in range(gcfg.nh):
                    c, h = float(gcash[i]), float(ghouse[j])
                    pay_cand = []
                    pay_cand.append(_solve_one_state_continuous(c, h, aux_pay, h * (1 - fp.adjcost - ppt) + c - otcost - ppcost, "buy", True, fp, minhouse2, model_pay_np))
                    pay_cand.append(_solve_one_state_continuous(c, h, aux_pay, h * (1 - fp.adjcost - ppt) + c - otcost - ppcost, "zero", True, fp, minhouse2, model_pay_np))
                    pay_cand.append(_solve_one_state_continuous(c, h, aux_pay, h * (1 - fp.adjcost - ppt) + c, "buy", False, fp, minhouse2, model_pay_np))
                    pay_cand.append(_solve_one_state_continuous(c, h, aux_pay, h * (1 - fp.adjcost - ppt) + c, "zero", False, fp, minhouse2, model_pay_np))
                    pay_cand.append(_solve_one_state_continuous(c, h, aux_pay, h * (-ppt) + c - otcost - ppcost, "keep", True, fp, minhouse2, model_pay_np))
                    pay_cand.append(_solve_one_state_continuous(c, h, aux_pay, h * (-ppt) + c, "keep", False, fp, minhouse2, model_pay_np))
                    pay_best = max(pay_cand, key=lambda z: z[3])

                    nopay_cand = []
                    nopay_cand.append(_solve_one_state_continuous(c, h, aux_nopay, h * (1 - fp.adjcost - ppt) + c, "buy", False, fp, minhouse2, model_nopay_np))
                    nopay_cand.append(_solve_one_state_continuous(c, h, aux_nopay, h * (1 - fp.adjcost - ppt) + c, "zero", False, fp, minhouse2, model_nopay_np))
                    nopay_cand.append(_solve_one_state_continuous(c, h, aux_nopay, h * (-ppt) + c, "keep", False, fp, minhouse2, model_nopay_np))
                    nopay_best = max(nopay_cand, key=lambda z: z[3])

                    best_np[i, j, :] = np.asarray(pay_best if pay_best[3] >= nopay_best[3] else nopay_best, dtype=float)
        else:
            pay_stack = jnp.stack(
                [
                    _solve_case_batch(aux_pay, ppcost, otcost, minhouse2, h_mode="buy", can_participate=True, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c - otc - ppc),
                    _solve_case_batch(aux_pay, ppcost, otcost, minhouse2, h_mode="zero", can_participate=True, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c - otc - ppc),
                    _solve_case_batch(aux_pay, ppcost, otcost, minhouse2, h_mode="buy", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c),
                    _solve_case_batch(aux_pay, ppcost, otcost, minhouse2, h_mode="zero", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c),
                    _solve_case_batch(aux_pay, ppcost, otcost, minhouse2, h_mode="keep", can_participate=True, budget_fn=lambda c, h, ppc, otc: h * (-ppt) + c - otc - ppc),
                    _solve_case_batch(aux_pay, ppcost, otcost, minhouse2, h_mode="keep", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (-ppt) + c),
                ],
                axis=0,
            )
            pay_idx = jnp.argmax(pay_stack[:, :, 3], axis=0)
            pay_best = jnp.take_along_axis(pay_stack, pay_idx[None, :, None], axis=0)[0]

            nopay_stack = jnp.stack(
                [
                    _solve_case_batch(aux_nopay, ppcost, 0.0, minhouse2, h_mode="buy", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c),
                    _solve_case_batch(aux_nopay, ppcost, 0.0, minhouse2, h_mode="zero", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (1 - fp.adjcost - ppt) + c),
                    _solve_case_batch(aux_nopay, ppcost, 0.0, minhouse2, h_mode="keep", can_participate=False, budget_fn=lambda c, h, ppc, otc: h * (-ppt) + c),
                ],
                axis=0,
            )
            nopay_idx = jnp.argmax(nopay_stack[:, :, 3], axis=0)
            nopay_best = jnp.take_along_axis(nopay_stack, nopay_idx[None, :, None], axis=0)[0]

            use_pay = pay_best[:, 3] >= nopay_best[:, 3]
            best = jnp.where(use_pay[:, None], pay_best, nopay_best)
            best_np = np.asarray(best).reshape(gcfg.ncash, gcfg.nh, 4)
        C1[:, :, t], A1[:, :, t], H1[:, :, t], V1[:, :, t] = best_np[:, :, 0], best_np[:, :, 1], best_np[:, :, 2], best_np[:, :, 3]

    # MATLAB 中对极小房产选择做清零
    H[H < 1e-3] = 0.0
    H1[H1 < 1e-3] = 0.0
    return C, A, H, C1, A1, H1
