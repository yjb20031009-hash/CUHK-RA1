"""Wrappers for MATLAB `my_estimation_prepostdid1*.m` family.

These functions reuse the engineering-first `my_estimation_prepost` backbone,
while applying variant-specific parameter normalization and sample filtering.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import numpy as np
from scipy.io import loadmat, savemat

from .my_estimation_prepost import EstimationConfig, my_estimation_prepost


def _normalize_params(myparam: np.ndarray, *, incaa: float, incb1: float, incb2: float, incb3: float, otcost_scale: float) -> tuple[np.ndarray, float | None, float | None]:
    """Convert MATLAB normalized parameters to level parameters.

    Returns:
      p: array([ppcost, otcost, rho, delta, psi]) in level form used by base estimator
      mu, muh: optional return premia overrides
    """
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
    """Create temp mat file preserving all keys but filtered mySample by col9 (index 8)."""
    mat = loadmat(sample_path)
    if "mySample" in mat and fl_filter is not None:
        ms = np.asarray(mat["mySample"])
        if ms.ndim == 2 and ms.shape[1] >= 9:
            mat["mySample"] = ms[ms[:, 8] == fl_filter]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mat")
    tmp.close()
    savemat(tmp.name, mat)
    return tmp.name


def _run_variant(
    myparam: np.ndarray,
    *,
    cfg: EstimationConfig,
    otcost_scale: float,
    fl_filter: int | None,
    sample_prepost_path: str,
    sim_sample_path: str,
    use_sim_data: bool,
    recompute_policy: bool,
):
    p5, mu, muh = _normalize_params(
        myparam,
        incaa=cfg.incaa,
        incb1=cfg.incb1,
        incb2=cfg.incb2,
        incb3=cfg.incb3,
        otcost_scale=otcost_scale,
    )

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
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def my_estimation_prepostdid1(
    myparam: np.ndarray,
    *,
    sample_prepost_path: str = "mySample_pre10.mat",
    sim_sample_path: str = "sim_mySample2.mat",
    use_sim_data: bool = False,
    recompute_policy: bool = True,
):
    """Full-sample DID1 wrapper."""
    cfg = replace(
        EstimationConfig(),
        ncash=21,
        nh=11,
        incaa=9.89959,
        incb1=0.0092466,
        incb2=-1.447669 / 1e4,
        incb3=0.0,
    )
    return _run_variant(
        myparam,
        cfg=cfg,
        otcost_scale=200000.0,
        fl_filter=None,
        sample_prepost_path=sample_prepost_path,
        sim_sample_path=sim_sample_path,
        use_sim_data=use_sim_data,
        recompute_policy=recompute_policy,
    )


def my_estimation_prepostdid1_high(
    myparam: np.ndarray,
    *,
    sample_prepost_path: str = "mySample_pre10.mat",
    sim_sample_path: str = "sim_mySample2.mat",
    use_sim_data: bool = False,
    recompute_policy: bool = True,
):
    """High-financial-literacy DID1 wrapper (mySample(:,9)==1)."""
    cfg = replace(
        EstimationConfig(),
        ncash=21,
        nh=11,
        incaa=9.88469,
        incb1=0.012571,
        incb2=-1.248147 / 1e4,
        incb3=0.0,
    )
    return _run_variant(
        myparam,
        cfg=cfg,
        otcost_scale=500000.0,
        fl_filter=1,
        sample_prepost_path=sample_prepost_path,
        sim_sample_path=sim_sample_path,
        use_sim_data=use_sim_data,
        recompute_policy=recompute_policy,
    )


def my_estimation_prepostdid1_low(
    myparam: np.ndarray,
    *,
    sample_prepost_path: str = "mySample_pre10.mat",
    sim_sample_path: str = "sim_mySample2.mat",
    use_sim_data: bool = False,
    recompute_policy: bool = True,
):
    """Low-financial-literacy DID1 wrapper (mySample(:,9)==0)."""
    cfg = replace(
        EstimationConfig(),
        ncash=21,
        nh=11,
        incaa=9.87492,
        incb1=0.0096951,
        incb2=-1.81387 / 1e4,
        incb3=0.0,
    )
    return _run_variant(
        myparam,
        cfg=cfg,
        otcost_scale=200000.0,
        fl_filter=0,
        sample_prepost_path=sample_prepost_path,
        sim_sample_path=sim_sample_path,
        use_sim_data=use_sim_data,
        recompute_policy=recompute_policy,
    )
