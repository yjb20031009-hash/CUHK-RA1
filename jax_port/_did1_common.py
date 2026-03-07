from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import numpy as np
from scipy.io import loadmat, savemat

from .my_estimation_prepost import EstimationConfig, my_estimation_prepost


def normalize_params(
    myparam: np.ndarray,
    *,
    incaa: float,
    incb1: float,
    incb2: float,
    incb3: float,
    otcost_scale: float,
) -> tuple[np.ndarray, float | None, float | None]:
    p = np.asarray(myparam, dtype=float).reshape(-1).copy()
    if p.size < 5:
        raise ValueError("myparam must have at least 5 elements")

    y60 = np.exp(incaa + incb1 * 60 + incb2 * 60**2 + incb3 * 60**3)
    y61 = np.exp(incaa + incb1 * 61 + incb2 * 61**2 + incb3 * 61**3)

    if p[1] < 1.0:
        rho = p[2] * 10.0 + 2.0
        delta = p[3] * 0.29 + 0.70
        psi = p[4] * 0.40 + 0.30
        ppcost = p[0] * 10000.0 / (y60 + y61)
        otcost = p[1] * otcost_scale / (y60 + y61)
        mu = p[5] * 0.20 if p.size > 5 else None
        muh = p[6] * 0.20 if p.size > 6 else None
    else:
        rho, delta, psi = p[2], p[3], p[4]
        ppcost = p[0] / (y60 + y61)
        otcost = p[1] / (y60 + y61)
        mu = p[5] if p.size > 5 else None
        muh = p[6] if p.size > 6 else None

    return np.array([ppcost, otcost, rho, delta, psi], dtype=float), mu, muh


def _with_filtered_sample(sample_path: str, fl_filter: int | None) -> str:
    mat = loadmat(sample_path)
    if "mySample" in mat and fl_filter is not None:
        ms = np.asarray(mat["mySample"])
        if ms.ndim == 2 and ms.shape[1] >= 9:
            mat["mySample"] = ms[ms[:, 8] == fl_filter]

    # scipy.io.savemat 会忽略以下划线开头的键（如 __header__/__version__/__globals__）并给出告警。
    # 这里先过滤掉这些元键，避免运行时反复出现 MatWriteWarning。
    mat_to_save = {k: v for k, v in mat.items() if not k.startswith("__")}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mat")
    tmp.close()
    savemat(tmp.name, mat_to_save)
    return tmp.name


def _default_moments_path(fl_filter: int | None) -> str:
    """Default DID moments file by variant, matching MATLAB wrappers."""
    if fl_filter == 1:
        return "Sample_did_nosample_high.mat"
    if fl_filter == 0:
        return "Sample_did_nosample_low.mat"
    return "Sample_did_nosample.mat"


def run_variant(
    myparam: np.ndarray,
    *,
    cfg: EstimationConfig,
    otcost_scale: float,
    fl_filter: int | None,
    sample_prepost_path: str,
    sim_sample_path: str,
    use_sim_data: bool,
    recompute_policy: bool,
    moments_path: str | None = None,
):
    p5, mu, muh = normalize_params(
        myparam,
        incaa=cfg.incaa,
        incb1=cfg.incb1,
        incb2=cfg.incb2,
        incb3=cfg.incb3,
        otcost_scale=otcost_scale,
    )
    # IMPORTANT: temporary filtered sample file usually contains only `mySample`,
    # so DID moments must come from a dedicated moments file.
    resolved_moments_path = moments_path or _default_moments_path(fl_filter)

    tmp = _with_filtered_sample(sample_prepost_path, fl_filter)
    try:
        return my_estimation_prepost(
            p5,
            cfg=cfg,
            mu=mu,
            muh=muh,
            use_sim_data=use_sim_data,
            recompute_policy=recompute_policy,
            sample_prepost_path=tmp,
            sim_sample_path=sim_sample_path,
            moments_path=resolved_moments_path,
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def base_cfg(**kwargs) -> EstimationConfig:
    return replace(EstimationConfig(), ncash=21, nh=11, **kwargs)
