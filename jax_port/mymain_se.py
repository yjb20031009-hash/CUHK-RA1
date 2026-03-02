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
        return interp2_regular(ghouse, gcash, v_next.T, housing_nn, cash_nn, method="linear", bounds="clip")

    return model_fn


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
) -> tuple[float, float, float, float]:
    """单状态点求解。

    给定预算 b 和模式（买房/清零/保持），在候选网格上枚举 (c,a,h)，
    调用 `my_auxv_cal` 计算目标值（其输出是 -V），再选最小目标值。
    返回 (c*, a*, h*, V*)。
    """
    if b < 0.25:
        # 预算过低视作不可行，返回极差价值
        return (0.25, 0.0, 0.0, -jnp.inf)

    cgrid = jnp.linspace(0.25, max(b, 0.25), gcfg.nc)
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
        return aux_params, ppcost, otcost, minhouse2

    # Loop 1: 已经支付过 one-time cost 的人群
    ppcost = float(ppcost_in)
    otcost = 0.0
    for t in range(tn - 2, -1, -1):
        aux, ppcost, otcost, minhouse2 = loop_block(V[:, :, t + 1], t, ppcost, otcost)
        for i in range(gcfg.ncash):
            for j in range(gcfg.nh):
                cash, house = float(gcash[i]), float(ghouse[j])

                # 对应 MATLAB 的 6 种 case：
                # 1) 处置+参与+买房 2) 处置+参与+不买房
                # 3) 处置+不参与+买房 4) 处置+不参与+不买房
                # 5) 不处置+参与      6) 不处置+不参与
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

    # Loop 2: 未支付过 one-time cost 的人群
    # 对比“当期支付 otcost”与“继续不支付”两条路径，选价值更高者
    ppcost = float(ppcost_in)
    otcost = float(otcost_in)
    for t in range(tn - 2, -1, -1):
        aux_pay, ppcost, otcost, minhouse2 = loop_block(V[:, :, t + 1], t, ppcost, otcost)
        aux_nopay = AuxVParams(
            **{
                **aux_pay.__dict__,
                "otcost": 0.0,
                "model_fn": _build_model_fn(jnp.asarray(V1[:, :, t + 1]), gcash, ghouse),
            }
        )

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
                # 不支付路径：按 MATLAB 逻辑仅考虑不参与股票相关 case
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

    # MATLAB 中对极小房产选择做清零
    H[H < 1e-3] = 0.0
    H1[H1 < 1e-3] = 0.0
    return C, A, H, C1, A1, H1
