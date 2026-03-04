"""Engineering-first Python port of MATLAB `my_estimation_prepost.m`.

This module mirrors the original workflow:
- build/scale model objects
- solve pre/post policy functions via `mymain_se`
- simulate one-step transitions pre/post
- run per-shock OLS and probability-weighted averaging
- return (ggvalue, gvalue, betamat)
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat

from .interp2 import interp2_regular
from .mymain_se import FixedParams, GridCfg, mymain_se
from .tauchen_hussey import tauchen_hussey


@dataclass(frozen=True)
class EstimationConfig:
    stept: float = 2.0
    tb_year: float = 20.0
    tr_year: float = 62.0
    td_year: float = 100.0

    ncash: int = 11
    nh: int = 6
    maxcash: float = 40.0
    mincash: float = 0.25
    maxhouse: float = 40.0
    minhouse: float = 0.0

    adjcost: float = 0.07
    ret_fac: float = 0.6
    minhouse2_value: float = 0.0

    ppt_pre: float = 0.0
    ppt_post: float = 0.008

    n_shock_1d: int = 3
    corr_hs: float = -0.08
    r: float = 1.05 - 0.048
    mu: float = 0.08
    muh: float = 0.08
    sigr: float = 0.42
    sigrh: float = 0.28

    incaa: float = 0.0
    incb1: float = 0.0
    incb2: float = 0.0
    incb3: float = 0.0

    pfun_pre_path: str = "PFunction_prepostdid1_pre.mat"
    pfun_post_path: str = "PFunction_prepostdid1_post.mat"


def _compute_tn(cfg: EstimationConfig) -> int:
    tb = int(cfg.tb_year / cfg.stept)
    td = int(cfg.td_year / cfg.stept)
    return td - tb + 1


def build_state_grids(cfg: EstimationConfig) -> tuple[np.ndarray, np.ndarray]:
    gcash = np.exp(np.linspace(np.log(cfg.mincash), np.log(cfg.maxcash), cfg.ncash))
    ghouse = np.exp(np.linspace(np.log(cfg.minhouse + 1.0), np.log(cfg.maxhouse + 1.0), cfg.nh)) - 1.0
    return gcash.reshape(-1, 1), ghouse.reshape(-1, 1)


def compute_gyp_path(cfg: EstimationConfig) -> np.ndarray:
    tn = _compute_tn(cfg)
    tb = int(cfg.tb_year / cfg.stept)
    tr = int(cfg.tr_year / cfg.stept)

    gyp = np.zeros(tn, dtype=np.float64)
    for t_mat in range(1, tn + 1):
        if t_mat >= (tr - tb + 1):
            gyp[t_mat - 1] = 1.0
        else:
            a1 = cfg.stept * (t_mat + tb - 1)
            a2 = cfg.stept * (t_mat + tb)

            def f(age: float) -> float:
                return np.exp(cfg.incaa + cfg.incb1 * age + cfg.incb2 * age**2 + cfg.incb3 * age**3)

            gyp[t_mat - 1] = np.exp((f(a2) + f(a2 - 1.0)) / (f(a1) + f(a1 - 1.0)) - 1.0)
    return gyp


def scale_backward(base: float, gyp: np.ndarray) -> np.ndarray:
    tn = gyp.shape[0]
    x = np.zeros(tn, dtype=np.float64)
    x[tn - 1] = float(base)
    for t in range(tn - 2, -1, -1):
        x[t] = x[t + 1] * gyp[t]
    return x


def build_return_process(cfg: EstimationConfig, tauchen_fn: Callable[..., tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    n = cfg.n_shock_1d
    grid, weig2 = tauchen_fn(n, 0, 0, 1, 1)
    grid = np.asarray(grid).reshape(n, 1)
    weig = np.asarray(weig2)[0, :].reshape(n, 1)

    gret = np.zeros((n, 1))
    for i in range(n):
        gret[i, 0] = cfg.r + cfg.mu + grid[i, 0] * cfg.sigr

    greth = np.zeros((n, n))
    for i in range(n):
        grid2 = grid[i, 0] * cfg.corr_hs + grid[:, 0] * np.sqrt(1 - cfg.corr_hs**2)
        greth[:, i] = cfg.r + cfg.muh + grid2 * cfg.sigrh

    greth_vec = greth.reshape(n * n, 1)
    nn = n * n
    gret_sh = np.zeros((nn, 3))
    for i in range(nn):
        gret_sh[i, 0] = gret[int(np.ceil((i + 1) / n)) - 1, 0]
        gret_sh[i, 1] = greth_vec[i, 0]
        gret_sh[i, 2] = float(weig[int(np.ceil((i + 1) / n)) - 1, 0] * weig[(i % n), 0])
    return gret_sh


def _interp_policy_scalar(gcash: np.ndarray, ghouse: np.ndarray, p: np.ndarray, house: float, cash: float, mode: str = "linear") -> float:
    out = interp2_regular(
        jnp.asarray(ghouse.reshape(-1)),
        jnp.asarray(gcash.reshape(-1)),
        jnp.asarray(p),
        jnp.asarray(house),
        jnp.asarray(cash),
        method="nearest" if mode == "nearest" else "linear",
        bounds="clip",
    )
    return float(out)


def simulate_one_step(*, t_mat: int, initial_ipart: int, cash0: float, house0: float, gcash: np.ndarray, ghouse: np.ndarray, C: np.ndarray, A: np.ndarray, H: np.ndarray, C1: np.ndarray, A1: np.ndarray, H1: np.ndarray, ppt: float, adjcost: float, otcost_t: float, ppcost_t: float, minhouse2_t: float, gyp_t: float, gret_sh: np.ndarray, r: float) -> dict[str, Any]:
    nn = gret_sh.shape[0]
    t0 = t_mat - 1

    if initial_ipart == 0:
        simC = _interp_policy_scalar(gcash, ghouse, C1[:, :, t0], house0, cash0, "linear")
        simA = _interp_policy_scalar(gcash, ghouse, A1[:, :, t0], house0, cash0, "linear")
        simH = _interp_policy_scalar(gcash, ghouse, H1[:, :, t0], house0, cash0, "linear")
        simAa = _interp_policy_scalar(gcash, ghouse, A1[:, :, t0], house0, cash0, "nearest")
    else:
        simC = _interp_policy_scalar(gcash, ghouse, C[:, :, t0], house0, cash0, "linear")
        simA = _interp_policy_scalar(gcash, ghouse, A[:, :, t0], house0, cash0, "linear")
        simH = _interp_policy_scalar(gcash, ghouse, H[:, :, t0], house0, cash0, "linear")
        simAa = _interp_policy_scalar(gcash, ghouse, A[:, :, t0], house0, cash0, "nearest")

    if simAa == 0.0:
        simA = 0.0

    simI = 1 if ((initial_ipart == 0 and simA > 0) or initial_ipart == 1) else 0

    if abs(simH - house0) <= 0.05 * house0 or (house0 == 0.0 and simH < minhouse2_t * 0.9):
        simH = house0
    if house0 == 0.0 and simH < minhouse2_t and simH >= minhouse2_t * 0.9:
        simH = minhouse2_t

    con1 = 1.0 if simH == house0 else 0.0
    con2 = 1.0 if (simI - initial_ipart) == 1 else 0.0
    con3 = 1.0 if simA > 0 else 0.0

    simS = cash0 + house0 * (1 - ppt - (1 - con1) * adjcost) - simC - simH - con2 * otcost_t - con3 * ppcost_t

    stock_ret = gret_sh[:, 0]
    house_gross = gret_sh[:, 1]
    simW = (simS * simA * stock_ret + simS * (1 - simA) * r * np.ones(nn)) / gyp_t
    simH2 = (simH * house_gross) / gyp_t

    simW = np.minimum(simW, 40.0)
    simH2 = np.minimum(simH2, 40.0)
    return {"simW": simW, "simH2": simH2, "simI": simI}


def build_regressors(mySample: np.ndarray, simW: np.ndarray, simH2: np.ndarray, simI: np.ndarray, nn: int, did_mode: bool) -> np.ndarray:
    l = mySample.shape[0]
    age_term = mySample[:, 2] + 2
    Xs = []
    for k in range(nn):
        if not did_mode:
            X = np.column_stack([np.ones(l), age_term, age_term**2, simW[:, k], simW[:, k] ** 2, simH2[:, k], simH2[:, k] ** 2, simI[:, 0], mySample[:, 6], mySample[:, 7]])
        else:
            X = np.column_stack([np.ones(l), age_term, age_term**2, simW[:, k], simW[:, k] ** 2, simH2[:, k], simH2[:, k] ** 2, simI[:, 0], np.zeros(l), np.ones(l), np.zeros(l)])
        Xs.append(X)
    return np.stack(Xs, axis=0)


def ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(X.T @ X, X.T @ y)


def _fallback_did_moments(sample_prepost_path: str) -> dict[str, Any] | None:
    """Try MATLAB-style DID moments files when current mat has only mySample."""
    base_dir = Path(sample_prepost_path).resolve().parent
    candidates = [
        base_dir / "Sample_did_nosample.mat",
        base_dir / "Sample_did_nosample_high.mat",
        base_dir / "Sample_did_nosample_low.mat",
        Path("Sample_did_nosample.mat"),
        Path("Sample_did_nosample_high.mat"),
        Path("Sample_did_nosample_low.mat"),
    ]
    for c in candidates:
        if not c.exists():
            continue
        m = loadmat(str(c))
        if m.get("W") is None:
            continue
        if m.get("beta1") is None or m.get("beta2") is None or m.get("beta3") is None or m.get("beta4") is None:
            continue
        return m
    return None


def my_estimation_prepost(
    myparam: np.ndarray,
    *,
    cfg: EstimationConfig = EstimationConfig(),
    tauchen_fn: Callable[..., tuple[np.ndarray, np.ndarray]] = tauchen_hussey,
    mymain_se_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = mymain_se,
    mu: float | None = None,
    muh: float | None = None,
    use_sim_data: bool = False,
    recompute_policy: bool = True,
    sample_prepost_path: str = "Sample_prepost.mat",
    sim_sample_path: str = "sim_mySample2.mat",
    moments_path: str | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    # === MATLAB section: 参数拆包 (ppcost, otcost, rho, delta, psi) ===
    myparam = np.asarray(myparam).reshape(-1)
    ppcost, otcost, rho, delta, psi = map(float, myparam[:5])

    if mu is not None:
        cfg = replace(cfg, mu=float(mu))
    if muh is not None:
        cfg = replace(cfg, muh=float(muh))

    # === MATLAB section: 载入样本 ===
    # 对应 my_estimation_prepost.m 中:
    #   - 横截面: load Sample_prepost.mat
    #   - DID/sim: load sim_mySample2.mat; mySample = sim_mySample
    mat = loadmat(sim_sample_path if use_sim_data else sample_prepost_path)
    mySample = np.asarray(mat.get("sim_mySample", mat.get("mySample")))

    # MATLAB mapping:
    # - 本函数主体对应 my_estimation_prepost.m 主分支：矩一般来自当前 load 的样本文件
    # - DID1 系列（my_estimation_prepostdid1*）在各自外层函数中显式 load Sample_did_nosample*.mat
    #   因此这里不再硬编码默认 DID moments 文件，避免与原函数语义混淆。
    moments_mat = loadmat(moments_path) if moments_path is not None else mat
    W = moments_mat.get("W")
    if W is None and moments_path is None:
        # Robust fallback: DID wrappers often pass a temporary sample-only mat.
        fallback = _fallback_did_moments(sample_prepost_path)
        if fallback is not None:
            moments_mat = fallback
            W = moments_mat.get("W")
    beta_blocks: list[np.ndarray] = []
    i = 1
    while True:
        bi = moments_mat.get(f"beta{i}")
        if bi is None:
            break
        beta_blocks.append(np.asarray(bi).reshape(-1))
        i += 1

    if W is None or len(beta_blocks) < 4:
        src = moments_path if moments_path is not None else (sim_sample_path if use_sim_data else sample_prepost_path)
        keys = sorted([k for k in moments_mat.keys() if not k.startswith("__")])
        hint = " (DID路径请显式传 moments_path，或在工作目录放置 Sample_did_nosample*.mat)" if moments_path is None else ""
        raise KeyError(
            f"moments file '{src}' must contain W and at least beta1..beta4; available keys={keys}{hint}"
        )

    beta_real = np.concatenate(beta_blocks, axis=0)

    # === MATLAB section: 打状态网格 (gcash, ghouse) ===
    gcash, ghouse = build_state_grids(cfg)
    tb = int(cfg.tb_year / cfg.stept)
    tr = int(cfg.tr_year / cfg.stept)

    # === MATLAB section: Tauchen-Hussey + gret_sh 构造 ===
    gret_sh = build_return_process(cfg, tauchen_fn)
    nn = gret_sh.shape[0]

    # === MATLAB section: 收入增长路径 gyp 与成本回推 ===
    gyp = compute_gyp_path(cfg)
    otcost_t = scale_backward(otcost, gyp)
    ppcost_t = scale_backward(ppcost, gyp)
    minhouse2_t = scale_backward(0.0, gyp)

    # === MATLAB section: 求/载入 policy function ===
    # 对应 PFunction_prepostdid1_pre.mat / PFunction_prepostdid1_post.mat 逻辑
    gcfg_for_solver = GridCfg(n=cfg.n_shock_1d, ncash=cfg.ncash, nh=cfg.nh)
    fp_for_solver = FixedParams(
        adjcost=cfg.adjcost,
        maxhouse=cfg.maxhouse,
        minhouse=cfg.minhouse,
        maxcash=cfg.maxcash,
        mincash=cfg.mincash,
        corr_hs=cfg.corr_hs,
        r=cfg.r,
        sigr=cfg.sigr,
        sigrh=cfg.sigrh,
        incaa=cfg.incaa,
        incb1=cfg.incb1,
        incb2=cfg.incb2,
        incb3=cfg.incb3,
        ret_fac=cfg.ret_fac,
        minhouse2_value=cfg.minhouse2_value,
    )

    def _coerce_policy_shape(arr: np.ndarray) -> np.ndarray:
        # expected: (ncash, nh, tn)
        if arr.ndim != 3:
            raise ValueError(f"policy array must be 3D, got shape={arr.shape}")
        if arr.shape[0] == cfg.ncash and arr.shape[1] == cfg.nh:
            return arr
        if arr.shape[0] == cfg.nh and arr.shape[1] == cfg.ncash:
            return np.transpose(arr, (1, 0, 2))
        raise ValueError(f"policy array has incompatible shape={arr.shape}, expected (ncash={cfg.ncash}, nh={cfg.nh}, tn)")

    def load_or_solve_policy(ppt: float, path: str):
        if recompute_policy or (not os.path.exists(path)):
            C, A, H, C1, A1, H1 = mymain_se_fn(
                ppt,
                ppcost,
                otcost,
                rho,
                delta,
                psi,
                cfg.mu,
                cfg.muh,
                gcfg=gcfg_for_solver,
                fp=fp_for_solver,
            )
            C, A, H = map(_coerce_policy_shape, (np.asarray(C), np.asarray(A), np.asarray(H)))
            C1, A1, H1 = map(_coerce_policy_shape, (np.asarray(C1), np.asarray(A1), np.asarray(H1)))
            savemat(path, {"C": C, "A": A, "H": H, "C1": C1, "A1": A1, "H1": H1})
            return C, A, H, C1, A1, H1
        pmat = loadmat(path)
        C, A, H = map(_coerce_policy_shape, (np.asarray(pmat["C"]), np.asarray(pmat["A"]), np.asarray(pmat["H"])))
        C1, A1, H1 = map(_coerce_policy_shape, (np.asarray(pmat["C1"]), np.asarray(pmat["A1"]), np.asarray(pmat["H1"])))
        return C, A, H, C1, A1, H1

    # === MATLAB section: 根据 policy + 样本做一期仿真 ===
    # 对应 my_estimation_prepost.m 中 pre/post 两段 simulation 主体
    def simulate_block(ppt: float, C: np.ndarray, A: np.ndarray, H: np.ndarray, C1: np.ndarray, A1: np.ndarray, H1: np.ndarray, sample_block: np.ndarray):
        l = sample_block.shape[0]
        initialW = sample_block[:, 3]
        initialH = sample_block[:, 4]
        initialAge = np.clip(np.floor((sample_block[:, 2] - 20) / 2).astype(int) + 1, 1, 40)
        initialIpart = sample_block[:, 5].astype(int)

        simW = np.zeros((l, nn))
        simH2 = np.zeros((l, nn))
        simI = np.zeros((l, 1), dtype=int)
        simCp1 = np.zeros((l, nn))
        simAp1 = np.zeros((l, nn))

        for i in range(l):
            t_mat = int(initialAge[i])
            cash0 = float(initialW[i] + (cfg.ret_fac if (t_mat > (tr - tb)) else 1.0))
            house0 = float(initialH[i])
            out = simulate_one_step(
                t_mat=t_mat,
                initial_ipart=int(initialIpart[i]),
                cash0=cash0,
                house0=house0,
                gcash=gcash,
                ghouse=ghouse,
                C=C,
                A=A,
                H=H,
                C1=C1,
                A1=A1,
                H1=H1,
                ppt=ppt,
                adjcost=cfg.adjcost,
                otcost_t=float(otcost_t[t_mat - 1]),
                ppcost_t=float(ppcost_t[t_mat - 1]),
                minhouse2_t=float(minhouse2_t[t_mat - 1]),
                gyp_t=float(gyp[t_mat - 1]),
                gret_sh=gret_sh,
                r=cfg.r,
            )
            simW[i, :], simH2[i, :], simI[i, 0] = out["simW"], out["simH2"], out["simI"]

            t1_mat = min(int(initialAge[i] + 1), C.shape[2])
            cash1 = simW[i, :] + (cfg.ret_fac if (t1_mat > (tr - tb)) else 1.0)
            house1 = simH2[i, :]

            # 向量化 shock 维度插值，避免逐 k Python 循环带来的 host/device 开销
            if simI[i, 0] == 0:
                simCp1_i = interp2_regular(
                    jnp.asarray(ghouse.reshape(-1)),
                    jnp.asarray(gcash.reshape(-1)),
                    jnp.asarray(C1[:, :, t1_mat - 1]),
                    jnp.asarray(house1),
                    jnp.asarray(cash1),
                    method="linear",
                    bounds="clip",
                )
                simAp1_i = interp2_regular(
                    jnp.asarray(ghouse.reshape(-1)),
                    jnp.asarray(gcash.reshape(-1)),
                    jnp.asarray(A1[:, :, t1_mat - 1]),
                    jnp.asarray(house1),
                    jnp.asarray(cash1),
                    method="linear",
                    bounds="clip",
                )
                simAa_i = interp2_regular(
                    jnp.asarray(ghouse.reshape(-1)),
                    jnp.asarray(gcash.reshape(-1)),
                    jnp.asarray(A1[:, :, t1_mat - 1]),
                    jnp.asarray(house1),
                    jnp.asarray(cash1),
                    method="nearest",
                    bounds="clip",
                )
            else:
                simCp1_i = interp2_regular(
                    jnp.asarray(ghouse.reshape(-1)),
                    jnp.asarray(gcash.reshape(-1)),
                    jnp.asarray(C[:, :, t1_mat - 1]),
                    jnp.asarray(house1),
                    jnp.asarray(cash1),
                    method="linear",
                    bounds="clip",
                )
                simAp1_i = interp2_regular(
                    jnp.asarray(ghouse.reshape(-1)),
                    jnp.asarray(gcash.reshape(-1)),
                    jnp.asarray(A[:, :, t1_mat - 1]),
                    jnp.asarray(house1),
                    jnp.asarray(cash1),
                    method="linear",
                    bounds="clip",
                )
                simAa_i = interp2_regular(
                    jnp.asarray(ghouse.reshape(-1)),
                    jnp.asarray(gcash.reshape(-1)),
                    jnp.asarray(A[:, :, t1_mat - 1]),
                    jnp.asarray(house1),
                    jnp.asarray(cash1),
                    method="nearest",
                    bounds="clip",
                )

            simCp1[i, :] = np.asarray(simCp1_i, dtype=float)
            simAp1_i = np.asarray(simAp1_i, dtype=float)
            simAa_i = np.asarray(simAa_i, dtype=float)
            simAp1[i, :] = np.where(simAa_i == 0.0, 0.0, simAp1_i)

        return simW, simH2, simI, simCp1, simAp1

    # === MATLAB section: pre-tax (ppt=0.00) ===
    C, A, H, C1, A1, H1 = load_or_solve_policy(cfg.ppt_pre, cfg.pfun_pre_path)
    pre_mask = (mySample[:, 7] == 0) | (mySample[:, 6] == 0) if not use_sim_data else (mySample[:, 6] == 0)
    mySample1 = mySample[pre_mask]
    simW, simH2, simI, simCp1, simAp1 = simulate_block(cfg.ppt_pre, C, A, H, C1, A1, H1, mySample1)
    X_pre = build_regressors(mySample1, simW, simH2, simI, nn, did_mode=False)

    # === MATLAB section: post-tax (ppt=0.008) ===
    C, A, H, C1, A1, H1 = load_or_solve_policy(cfg.ppt_post, cfg.pfun_post_path)
    post_mask = (mySample[:, 7] == 1) & (mySample[:, 6] == 1) if not use_sim_data else (mySample[:, 6] == 1)
    mySample2 = mySample[post_mask]
    simW2, simH2_2, simI2, simCp1_2, simAp1_2 = simulate_block(cfg.ppt_post, C, A, H, C1, A1, H1, mySample2)
    X_post = build_regressors(mySample2, simW2, simH2_2, simI2, nn, did_mode=False)

    # === MATLAB section: 分 shock 回归 + 概率加权平均 ===
    def shock_avg_beta(Xk: np.ndarray, yk: np.ndarray) -> np.ndarray:
        betas = np.stack([ols_beta(Xk[k], yk[:, k]) for k in range(nn)], axis=1)
        return (betas @ gret_sh[:, 2].reshape(nn, 1)).reshape(-1)

    beta_sim_all = np.concatenate(
        [
            shock_avg_beta(X_pre, simCp1),
            shock_avg_beta(X_pre, (simAp1 > 0).astype(np.float64)),
            shock_avg_beta(X_pre, simAp1),
            shock_avg_beta(X_post, simCp1_2),
            shock_avg_beta(X_post, (simAp1_2 > 0).astype(np.float64)),
            shock_avg_beta(X_post, simAp1_2),
        ],
        axis=0,
    )
    beta_sim = beta_sim_all[: beta_real.shape[0]]

    # === MATLAB section: 目标函数输出 ===
    # betamat = [empirical moments, model-implied moments]
    # gvalue  = empirical - model
    # ggvalue = g' W g
    gvalue = beta_real - beta_sim
    ggvalue = float(gvalue.T @ np.asarray(W) @ gvalue)
    betamat = np.column_stack([beta_real, beta_sim])
    return ggvalue, gvalue, betamat
