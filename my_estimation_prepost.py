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

import numpy as np
from scipy.io import loadmat, savemat

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
    # Supported modes in `mymain_se`: gpu/discrete/continuous/continuous2/gpu_continuous.
    solver_mode: str = "gpu"
    discrete_na: int = 9
    discrete_nc: int = 5
    discrete_nh2: int = 9
    continuous_maxiter: int = 80
    continuous_ftol: float = 1e-6
    continuous_constraint_tol: float | None = 1e-2
    interp_method: str = "linear"
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


def _interp_policy_vector_np(
    gcash: np.ndarray,
    ghouse: np.ndarray,
    p: np.ndarray,
    house: np.ndarray,
    cash: np.ndarray,
    mode: str = "linear",
) -> np.ndarray:
    """Fast NumPy interpolation on regular grid for 1D query vectors."""
    xg = np.asarray(ghouse, dtype=float).reshape(-1)
    yg = np.asarray(gcash, dtype=float).reshape(-1)
    v = np.asarray(p, dtype=float)

    xq = np.asarray(house, dtype=float).reshape(-1)
    yq = np.asarray(cash, dtype=float).reshape(-1)

    xq = np.clip(xq, xg[0], xg[-1])
    yq = np.clip(yq, yg[0], yg[-1])

    if mode == "nearest":
        ix_r = np.clip(np.searchsorted(xg, xq, side="left"), 0, xg.size - 1)
        iy_r = np.clip(np.searchsorted(yg, yq, side="left"), 0, yg.size - 1)
        ix_l = np.maximum(ix_r - 1, 0)
        iy_l = np.maximum(iy_r - 1, 0)
        ix = np.where(np.abs(xq - xg[ix_l]) <= np.abs(xq - xg[ix_r]), ix_l, ix_r)
        iy = np.where(np.abs(yq - yg[iy_l]) <= np.abs(yq - yg[iy_r]), iy_l, iy_r)
        return v[iy, ix]

    ix = np.clip(np.searchsorted(xg, xq, side="right") - 1, 0, xg.size - 2)
    iy = np.clip(np.searchsorted(yg, yq, side="right") - 1, 0, yg.size - 2)

    x0 = xg[ix]
    x1 = xg[ix + 1]
    y0 = yg[iy]
    y1 = yg[iy + 1]

    tx = np.where(x1 > x0, (xq - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 > y0, (yq - y0) / (y1 - y0), 0.0)

    v00 = v[iy, ix]
    v10 = v[iy, ix + 1]
    v01 = v[iy + 1, ix]
    v11 = v[iy + 1, ix + 1]

    v0 = v00 * (1.0 - tx) + v10 * tx
    v1 = v01 * (1.0 - tx) + v11 * tx
    return v0 * (1.0 - ty) + v1 * ty


def _interp_policy_scalar(gcash: np.ndarray, ghouse: np.ndarray, p: np.ndarray, house: float, cash: float, mode: str = "linear") -> float:
    return float(_interp_policy_vector_np(gcash, ghouse, p, np.array([house]), np.array([cash]), mode=mode)[0])


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
    """MATLAB-like OLS with robust fallbacks for ill-conditioned/singular inputs."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # MATLAB often proceeds with warnings on near-singular systems; emulate that behavior
    # by filtering non-finite rows and using progressively more stable fallbacks.
    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    Xf = X[finite_mask]
    yf = y[finite_mask]

    if Xf.size == 0 or Xf.shape[0] == 0:
        return np.zeros(X.shape[1], dtype=float)

    try:
        return np.linalg.pinv(Xf, rcond=1e-10) @ yf
    except np.linalg.LinAlgError:
        try:
            return np.linalg.lstsq(Xf, yf, rcond=1e-8)[0]
        except np.linalg.LinAlgError:
            xtx = Xf.T @ Xf
            xty = Xf.T @ yf
            lam = 1e-8
            return np.linalg.solve(xtx + lam * np.eye(xtx.shape[0]), xty)


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
    allowed_solver_modes = {"gpu", "discrete", "continuous", "continuous2", "gpu_continuous"}
    if cfg.solver_mode not in allowed_solver_modes:
        raise ValueError(
            f"cfg.solver_mode={cfg.solver_mode!r} is invalid; choose from {sorted(allowed_solver_modes)}"
        )

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
    gcfg_for_solver = GridCfg(
        n=cfg.n_shock_1d,
        ncash=cfg.ncash,
        nh=cfg.nh,
        na=cfg.discrete_na,
        nc=cfg.discrete_nc,
        nh2=cfg.discrete_nh2,
    )
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
                solver_mode=cfg.solver_mode,
                continuous_maxiter=cfg.continuous_maxiter,
                continuous_ftol=cfg.continuous_ftol,
                continuous_constraint_tol=cfg.continuous_constraint_tol,
                interp_method=cfg.interp_method,
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

        # 分层矢量化（第1层）：按 (initialIpart, t_mat) 分组批量插值，替代逐样本 simulate_one_step 调用
        t_vec = initialAge.astype(int)
        cash0_all = initialW + np.where(t_vec > (tr - tb), cfg.ret_fac, 1.0)
        house0_all = initialH

        simC = np.zeros(l, dtype=float)
        simA = np.zeros(l, dtype=float)
        simH = np.zeros(l, dtype=float)
        simAa = np.zeros(l, dtype=float)

        for ipart_val in (0, 1):
            ip_mask = (initialIpart == ipart_val)
            if not np.any(ip_mask):
                continue
            unique_t = np.unique(t_vec[ip_mask])
            for t_mat in unique_t:
                grp = ip_mask & (t_vec == t_mat)
                idx = np.where(grp)[0]
                if idx.size == 0:
                    continue

                t0 = int(t_mat - 1)
                c_pol = C1[:, :, t0] if ipart_val == 0 else C[:, :, t0]
                a_pol = A1[:, :, t0] if ipart_val == 0 else A[:, :, t0]
                h_pol = H1[:, :, t0] if ipart_val == 0 else H[:, :, t0]

                hq = house0_all[idx]
                cq = cash0_all[idx]

                simC[idx] = _interp_policy_vector_np(gcash, ghouse, c_pol, hq, cq, mode="linear")
                simA[idx] = _interp_policy_vector_np(gcash, ghouse, a_pol, hq, cq, mode="linear")
                simH[idx] = _interp_policy_vector_np(gcash, ghouse, h_pol, hq, cq, mode="linear")
                simAa[idx] = _interp_policy_vector_np(gcash, ghouse, a_pol, hq, cq, mode="nearest")

        simA = np.where(simAa == 0.0, 0.0, simA)
        simI[:, 0] = (((initialIpart == 0) & (simA > 0.0)) | (initialIpart == 1)).astype(int)

        minhouse2_vec = minhouse2_t[t_vec - 1]
        keep_house = (np.abs(simH - house0_all) <= 0.05 * house0_all) | ((house0_all == 0.0) & (simH < minhouse2_vec * 0.9))
        simH = np.where(keep_house, house0_all, simH)
        bump_min_house = (house0_all == 0.0) & (simH < minhouse2_vec) & (simH >= minhouse2_vec * 0.9)
        simH = np.where(bump_min_house, minhouse2_vec, simH)

        con1 = (simH == house0_all).astype(float)
        con2 = ((simI[:, 0] - initialIpart) == 1).astype(float)
        con3 = (simA > 0.0).astype(float)

        otcost_vec = otcost_t[t_vec - 1]
        ppcost_vec = ppcost_t[t_vec - 1]
        gyp_vec = gyp[t_vec - 1]

        simS = cash0_all + house0_all * (1.0 - ppt - (1.0 - con1) * cfg.adjcost) - simC - simH - con2 * otcost_vec - con3 * ppcost_vec

        stock_ret = gret_sh[:, 0]
        house_gross = gret_sh[:, 1]
        simW[:, :] = (simS[:, None] * simA[:, None] * stock_ret[None, :] + simS[:, None] * (1.0 - simA[:, None]) * cfg.r) / gyp_vec[:, None]
        simH2[:, :] = (simH[:, None] * house_gross[None, :]) / gyp_vec[:, None]

        simW[:, :] = np.minimum(simW, 40.0)
        simH2[:, :] = np.minimum(simH2, 40.0)

        # 批量做 t+1 插值：按 (t1_mat, simI) 分组，避免在 i 循环中频繁调用插值核
        t1_vec = np.minimum(initialAge + 1, C.shape[2]).astype(int)
        cash1_all = simW + np.where(t1_vec[:, None] > (tr - tb), cfg.ret_fac, 1.0)
        house1_all = simH2

        for ipart_val in (0, 1):
            ip_mask = (simI[:, 0] == ipart_val)
            if not np.any(ip_mask):
                continue
            unique_t1 = np.unique(t1_vec[ip_mask])
            for t1_mat in unique_t1:
                grp = ip_mask & (t1_vec == t1_mat)
                idx = np.where(grp)[0]
                if idx.size == 0:
                    continue

                c_pol = C1[:, :, t1_mat - 1] if ipart_val == 0 else C[:, :, t1_mat - 1]
                a_pol = A1[:, :, t1_mat - 1] if ipart_val == 0 else A[:, :, t1_mat - 1]

                hq = house1_all[idx, :].reshape(-1)
                cq = cash1_all[idx, :].reshape(-1)

                cp = _interp_policy_vector_np(gcash, ghouse, c_pol, hq, cq, mode="linear").reshape(idx.size, nn)
                ap = _interp_policy_vector_np(gcash, ghouse, a_pol, hq, cq, mode="linear").reshape(idx.size, nn)
                aa = _interp_policy_vector_np(gcash, ghouse, a_pol, hq, cq, mode="nearest").reshape(idx.size, nn)

                simCp1[idx, :] = cp
                simAp1[idx, :] = np.where(aa == 0.0, 0.0, ap)

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
