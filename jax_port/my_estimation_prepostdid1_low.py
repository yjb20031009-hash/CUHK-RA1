from __future__ import annotations

import numpy as np

from ._did1_common import base_cfg, run_variant


def my_estimation_prepostdid1_low(
    myparam: np.ndarray,
    *,
    sample_prepost_path: str = "mySample_pre10.mat",
    sim_sample_path: str = "sim_mySample2.mat",
    use_sim_data: bool = False,
    recompute_policy: bool = True,
):
    cfg = base_cfg(incaa=9.87492, incb1=0.0096951, incb2=-1.81387 / 1e4, incb3=0.0)
    return run_variant(
        myparam,
        cfg=cfg,
        otcost_scale=200000.0,
        fl_filter=0,
        sample_prepost_path=sample_prepost_path,
        sim_sample_path=sim_sample_path,
        use_sim_data=use_sim_data,
        recompute_policy=recompute_policy,
    )
