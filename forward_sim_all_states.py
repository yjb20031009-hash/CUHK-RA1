#!/usr/bin/env python3
"""Forward simulation over ALL state grid points and ALL periods.

Purpose
-------
Use saved policy functions (C/A/H or C1/A1/H1) and propagate a probability
mass over the full (cash, house) grid from t=0 to t=T-1.

This is intended as a diagnostic script: it lets you verify, period by period,
that policy-implied transitions are numerically stable and comparable to MATLAB.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat

from mymain_se import FixedParams, GridCfg, LifeCfg, _gret_sh, _income_growth
from tauchen_hussey import tauchen_hussey


def _pick_existing(path_candidates: Sequence[str]) -> str | None:
    for p in path_candidates:
        if Path(p).exists():
            return p
    return None


def _load_vec(mat: dict, *keys: str) -> np.ndarray:
    for k in keys:
        if k in mat:
            return np.asarray(mat[k], dtype=float).reshape(-1)
    raise KeyError(f"Cannot find any of keys={keys} in mat file")


def _load_3d(mat: dict, *keys: str) -> np.ndarray:
    for k in keys:
        if k in mat:
            arr = np.asarray(mat[k], dtype=float)
            arr = np.squeeze(arr)
            if arr.ndim != 3:
                raise ValueError(f"{k} must be 3D, got shape={arr.shape}")
            return arr
    raise KeyError(f"Cannot find any of keys={keys} in mat file")


def _nearest_idx(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    # x can be vector; return nearest grid index for each x.
    pos = np.searchsorted(grid, x)
    pos = np.clip(pos, 1, len(grid) - 1)
    left = grid[pos - 1]
    right = grid[pos]
    choose_right = (x - left) > (right - x)
    return np.where(choose_right, pos, pos - 1)


def _policy_triplet(mat: dict, loop_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if loop_mode == "loop2":
        return (
            _load_3d(mat, "C1", "C1_py", "C"),
            _load_3d(mat, "A1", "A1_py", "A"),
            _load_3d(mat, "H1", "H1_py", "H"),
        )
    return (
        _load_3d(mat, "C", "C_py"),
        _load_3d(mat, "A", "A_py"),
        _load_3d(mat, "H", "H_py"),
    )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Forward simulation for all states and all periods")
    p.add_argument("--policy-mat", default=None, help=".mat with policy arrays and grids")
    p.add_argument("--loop-mode", choices=["loop1", "loop2"], default="loop2", help="loop1 uses C/A/H; loop2 uses C1/A1/H1")
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--muh", type=float, default=0.08)
    p.add_argument("--ppt", type=float, default=0.0)
    p.add_argument("--ppcost", type=float, default=0.024689)
    p.add_argument("--otcost", type=float, default=0.097854)
    p.add_argument("--cash-min", type=float, default=0.25)
    p.add_argument("--cash-max", type=float, default=19.9)
    p.add_argument("--house-min", type=float, default=0.25)
    p.add_argument("--house-max", type=float, default=19.9)
    p.add_argument("--out-csv", default="forward_all_states_summary.csv")
    p.add_argument("--save-dist", default="", help="optional .npz path to save period distributions")
    args, _ = p.parse_known_args(argv)

    if args.policy_mat is None:
        args.policy_mat = _pick_existing((
            "fresh_jax_value_policy_gpu2.mat",
            "fresh_jax_value_policy_gpu.mat",
            "fresh_jax_value_policy_continuous.mat",
            "python_quick_test_result.mat",
        ))
    if args.policy_mat is None:
        raise SystemExit("Missing --policy-mat. Provide it explicitly or place a default policy mat in cwd.")

    mat = loadmat(args.policy_mat)
    gcash = _load_vec(mat, "gcash")
    ghouse = _load_vec(mat, "ghouse")
    C, A, H = _policy_triplet(mat, args.loop_mode)

    if C.shape != A.shape or C.shape != H.shape:
        raise ValueError(f"Policy shape mismatch: C={C.shape}, A={A.shape}, H={H.shape}")
    if C.shape[0] != len(gcash) or C.shape[1] != len(ghouse):
        raise ValueError("Policy first two dims must match gcash/ghouse lengths")

    ncash, nh, tn = C.shape

    fp = FixedParams()
    gcfg = GridCfg(ncash=ncash, nh=nh)
    lcfg = LifeCfg()

    grid, weig2 = tauchen_hussey(gcfg.n, 0.0, 0.0, 1.0, 1.0)
    grid = np.asarray(grid).reshape(-1)
    weig = np.asarray(weig2)[0, :].reshape(-1)
    gret = np.asarray(_gret_sh(fp, grid, weig, float(args.mu), float(args.muh), gcfg.n), dtype=float)
    stock_ret = gret[:, 0]
    house_gross = gret[:, 1]
    shock_prob = gret[:, 2]
    nshock = len(shock_prob)

    # Initial distribution: equal mass on all state points (tests all states).
    dist = np.full((ncash, nh), 1.0 / (ncash * nh), dtype=float)

    summaries: list[dict[str, float | int]] = []
    dist_hist = [dist.copy()]

    for t in range(tn):
        cash_state = np.broadcast_to(gcash[:, None], (ncash, nh))
        house_state = np.broadcast_to(ghouse[None, :], (ncash, nh))

        c_pol = np.asarray(C[:, :, t], dtype=float)
        a_pol = np.asarray(A[:, :, t], dtype=float)
        h_pol = np.asarray(H[:, :, t], dtype=float)

        income_t, gyp_t = _income_growth(fp, lcfg, t)
        ppc_t = float(args.ppcost) * float(gyp_t)
        otc_t = float(args.otcost) * float(gyp_t)

        # Mass-weighted policy moments at period t.
        mean_c = float(np.sum(dist * c_pol))
        mean_a = float(np.sum(dist * a_pol))
        mean_h_choice = float(np.sum(dist * h_pol))
        share_participate = float(np.sum(dist * (a_pol > 0.0)))
        share_adjust = float(np.sum(dist * (~np.isclose(h_pol, house_state, atol=1e-8, rtol=0.0))))

        next_dist = np.zeros_like(dist)

        for s in range(nshock):
            h_next = h_pol * house_gross[s] / gyp_t
            h_next = np.clip(h_next, float(args.house_min), float(args.house_max))

            adjust = ~np.isclose(h_pol, house_state, atol=1e-8, rtol=0.0)
            part = a_pol > 0.0

            sav = np.zeros_like(c_pol)
            # adjust + participate
            mask = adjust & part
            sav[mask] = cash_state[mask] + house_state[mask] * (1.0 - fp.adjcost - args.ppt) - c_pol[mask] - h_pol[mask] - ppc_t - otc_t
            # adjust + no participate
            mask = adjust & (~part)
            sav[mask] = cash_state[mask] + house_state[mask] * (1.0 - fp.adjcost - args.ppt) - c_pol[mask] - h_pol[mask]
            # keep + participate
            mask = (~adjust) & part
            sav[mask] = cash_state[mask] + house_state[mask] * (-args.ppt) - c_pol[mask] - ppc_t - otc_t
            # keep + no participate
            mask = (~adjust) & (~part)
            sav[mask] = cash_state[mask] + house_state[mask] * (-args.ppt) - c_pol[mask]

            cash_next = np.where(
                part,
                (sav * (1.0 - a_pol) * fp.r + sav * a_pol * stock_ret[s]) / gyp_t + income_t,
                sav * fp.r / gyp_t + income_t,
            )
            cash_next = np.clip(cash_next, float(args.cash_min), float(args.cash_max))

            ic = _nearest_idx(gcash, cash_next.ravel())
            ih = _nearest_idx(ghouse, h_next.ravel())

            w = (dist.ravel() * shock_prob[s])
            np.add.at(next_dist, (ic, ih), w)

        mass_sum = float(next_dist.sum())
        if mass_sum > 0:
            next_dist /= mass_sum

        mean_cash_next = float(np.sum(next_dist * gcash[:, None]))
        mean_house_next = float(np.sum(next_dist * ghouse[None, :]))

        summaries.append(
            {
                "t": int(t),
                "mass_sum": mass_sum,
                "mean_c": mean_c,
                "mean_a": mean_a,
                "mean_h_choice": mean_h_choice,
                "share_participate": share_participate,
                "share_adjust_house": share_adjust,
                "mean_cash_next": mean_cash_next,
                "mean_house_next": mean_house_next,
            }
        )

        dist = next_dist
        dist_hist.append(dist.copy())

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    print(f"[save] summary -> {out_csv.resolve()}")

    if args.save_dist:
        save_path = Path(args.save_dist)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            dist=np.asarray(dist_hist, dtype=float),  # shape=(tn+1, ncash, nh)
            gcash=np.asarray(gcash, dtype=float),
            ghouse=np.asarray(ghouse, dtype=float),
        )
        print(f"[save] distributions -> {save_path.resolve()}")


if __name__ == "__main__":
    main()
