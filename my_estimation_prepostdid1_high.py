from __future__ import annotations

import numpy as np

from ._did1_common import base_cfg, run_variant


def my_estimation_prepostdid1_high(
    myparam: np.ndarray,
    *,
    sample_prepost_path: str = "mySample_pre10.mat",
    sim_sample_path: str = "sim_mySample2.mat",
    use_sim_data: bool = False,
    recompute_policy: bool = True,
    moments_path: str = "Sample_did_nosample_high.mat",
    solver_mode: str | None = None,
    continuous_maxiter: int | None = None,
    continuous_ftol: float | None = None,
):
    cfg = base_cfg(incaa=9.88469, incb1=0.012571, incb2=-1.248147 / 1e4, incb3=0.0)
    return run_variant(
        myparam,
        cfg=cfg,
        otcost_scale=500000.0,
        fl_filter=1,
        sample_prepost_path=sample_prepost_path,
        sim_sample_path=sim_sample_path,
        use_sim_data=use_sim_data,
        recompute_policy=recompute_policy,
        moments_path=moments_path,
        solver_mode=solver_mode,
        continuous_maxiter=continuous_maxiter,
        continuous_ftol=continuous_ftol,
    )
