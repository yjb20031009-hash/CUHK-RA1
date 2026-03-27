#!/usr/bin/env python3
"""Decompose V1 candidates at T-1 into u_now / transitions / continuation terms.

This script re-solves the 9 loop-2 candidates for one selected state at t=T-1
(i.e., Python index t=tn-2), then decomposes each candidate into:
- u_now
- cash_nn (per shock)
- housing_nn (per shock)
- interp(V_next) (per shock)
- total V
- best case
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat

from my_auxv_cal import AuxVParams
from mymain_se import (
    FixedParams,
    GridCfg,
    LifeCfg,
    _build_model_fn_linear_np,
    _build_model_fn_spline,
    _gret_sh,
    _income_growth,
    _load_survprob,
    _minhouse2_normalized,
    _solve_one_state_gpu_continuous,
)
from tauchen_hussey import tauchen_hussey


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
                raise ValueError(f"{k} must be 3D, got {arr.shape}")
            return arr
    raise KeyError(f"Cannot find any of keys={keys} in mat file")


def _decompose_one(x: np.ndarray, aux: AuxVParams, thecash: float, thehouse: float, model_fn_np):
    myc, mya, myh = float(x[0]), float(x[1]), float(x[2])
    u_now = (1.0 - aux.delta) * (myc ** aux.psi_1)

    gret = np.asarray(aux.gret_sh, dtype=float)
    stock_ret = gret[:, 0]
    house_gross = gret[:, 1]
    weights = gret[:, 2]

    housing_nn = np.clip(myh * house_gross / aux.gyp, 0.25, 19.9)
    adjust_house = not np.isclose(myh, thehouse, atol=aux.eq_atol, rtol=0.0)
    participate = mya > 0.0

    if adjust_house and participate:
        sav = thecash + thehouse * (1.0 - aux.adjcost - aux.ppt) - myc - myh - aux.ppcost - aux.otcost
        cash_nn = (sav * (1.0 - mya) * aux.r + sav * mya * stock_ret) / aux.gyp + aux.income
    elif adjust_house and (not participate):
        sav = thecash + thehouse * (1.0 - aux.adjcost - aux.ppt) - myc - myh
        cash_nn = np.full_like(stock_ret, sav * aux.r / aux.gyp + aux.income)
    elif (not adjust_house) and participate:
        sav = thecash + thehouse * (-aux.ppt) - myc - aux.ppcost - aux.otcost
        cash_nn = (sav * (1.0 - mya) * aux.r + sav * mya * stock_ret) / aux.gyp + aux.income
    else:
        sav = thecash + thehouse * (-aux.ppt) - myc
        cash_nn = np.full_like(stock_ret, sav * aux.r / aux.gyp + aux.income)

    cash_nn = np.clip(cash_nn, 0.25, 19.9)
    interp_v = np.asarray(model_fn_np(housing_nn, cash_nn), dtype=float)
    interp_v = np.maximum(np.where(np.isfinite(interp_v), interp_v, 1e-8), 1e-8)

    surv = float(np.asarray(aux.survprob)[aux.t]) if np.asarray(aux.survprob).ndim == 1 else float(np.asarray(aux.survprob)[aux.t, 0])
    interp_pow = interp_v ** (1.0 - aux.rho)
    ev_term_raw = float(np.dot(weights, interp_pow) * surv)
    ev_term = max(ev_term_raw, 1e-8)
    total_core = u_now + aux.delta * (ev_term ** (1.0 / aux.theta))
    total_core = max(total_core, 1e-8)
    total_v = float(total_core ** aux.psi_2)

    return {
        "u_now": float(u_now),
        "surv": surv,
        "ev_term_raw": ev_term_raw,
        "ev_term": ev_term,
        "total_core": total_core,
        "total_v": total_v,
        "cash_nn": cash_nn,
        "housing_nn": housing_nn,
        "interp_v_next": interp_v,
        "interp_pow_1_minus_rho": interp_pow,
        "weighted_contrib": weights * interp_pow * surv,
        "weights": weights,
    }


def _pick_existing(path_candidates: Sequence[str]) -> str | None:
    for p in path_candidates:
        if Path(p).exists():
            return p
    return None


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Decompose T-1 V1 candidate cases for one state")
    p.add_argument("--policy-mat", default=None, help=".mat with V/V1 and optionally gcash/ghouse")
    p.add_argument("--surv-mat", default="surv.mat")
    p.add_argument("--cash-idx", type=int, default=None)
    p.add_argument("--house-idx", type=int, default=None)
    p.add_argument("--ppt", type=float, default=0.0, help="default follows MATLAB example block")
    p.add_argument("--ppcost", type=float, default=0.024689, help="base ppcost_in passed to mymain_se")
    p.add_argument("--otcost", type=float, default=0.097854, help="base otcost_in passed to mymain_se")
    p.add_argument("--rho", type=float, default=9.759, help="default follows MATLAB example block")
    p.add_argument("--delta", type=float, default=0.9871, help="default follows MATLAB example block")
    p.add_argument("--psi", type=float, default=0.67324, help="default follows MATLAB example block")
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--muh", type=float, default=0.08)
    p.add_argument("--interp-method", default="spline", choices=["linear", "nearest", "spline", "cubic"])
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--out-dir", default="tminus1_v1_candidate_decomp")
    args, _ = p.parse_known_args(argv)

    if args.policy_mat is None:
        args.policy_mat = _pick_existing((
            "fresh_jax_value_policy_gpu2.mat",
            "fresh_jax_value_policy_gpu.mat",
            "fresh_jax_value_policy_continuous.mat",
            "python_quick_test_result.mat",
        ))
    if args.policy_mat is None:
        raise SystemExit(
            "Missing --policy-mat. Provide it explicitly, or place a default policy mat file in current directory."
        )

    mat = loadmat(args.policy_mat)
    v = _load_3d(mat, "V", "V_py")
    v1 = _load_3d(mat, "V1", "V1_py")
    gcash = _load_vec(mat, "gcash")
    ghouse = _load_vec(mat, "ghouse")

    tn = v.shape[2]
    t = tn - 2  # T-1
    if t < 0:
        raise ValueError("Need at least 2 periods to analyze T-1")

    i = int(args.cash_idx) if args.cash_idx is not None else (len(gcash) - 1)
    j = int(args.house_idx) if args.house_idx is not None else (len(ghouse) - 1)
    if not (0 <= i < len(gcash) and 0 <= j < len(ghouse)):
        raise IndexError(f"state index out of range: cash_idx={i}, house_idx={j}")

    thecash = float(gcash[i])
    thehouse = float(ghouse[j])

    fp = FixedParams()
    gcfg = GridCfg(ncash=len(gcash), nh=len(ghouse))
    lcfg = LifeCfg()

    theta = (1.0 - args.rho) / (1.0 - 1.0 / args.psi)
    psi_1 = 1.0 - 1.0 / args.psi
    psi_2 = 1.0 / psi_1

    # T-1 paths for loop2 (V1): costs include one-time cost.
    income_t, gyp_t = _income_growth(fp, lcfg, t)
    ppc_t = float(args.ppcost) * float(gyp_t)
    otc_t = float(args.otcost) * float(gyp_t)
    minhouse2_t = float(_minhouse2_normalized(fp)) * float(gyp_t)

    survprob = _load_survprob(args.surv_mat)
    grid, weig2 = tauchen_hussey(gcfg.n, 0.0, 0.0, 1.0, 1.0)
    grid = np.asarray(grid).reshape(-1)
    weig = np.asarray(weig2)[0, :].reshape(-1)
    gret_sh = _gret_sh(fp, grid, weig, float(args.mu), float(args.muh), gcfg.n)

    v_next_pay = np.asarray(v[:, :, t + 1], dtype=float)
    v_next_nopay = np.asarray(v1[:, :, t + 1], dtype=float)

    if args.interp_method in {"spline", "cubic"}:
        model_pay = _build_model_fn_spline(v_next_pay, gcash, ghouse)
        model_nopay = _build_model_fn_spline(v_next_nopay, gcash, ghouse)
    else:
        model_pay = _build_model_fn_linear_np(v_next_pay, gcash, ghouse, method=args.interp_method)
        model_nopay = _build_model_fn_linear_np(v_next_nopay, gcash, ghouse, method=args.interp_method)

    aux_pay = AuxVParams(
        t=t,
        rho=float(args.rho),
        delta=float(args.delta),
        psi_1=float(psi_1),
        psi_2=float(psi_2),
        theta=float(theta),
        gyp=float(gyp_t),
        adjcost=float(fp.adjcost),
        ppt=float(args.ppt),
        ppcost=float(ppc_t),
        otcost=float(otc_t),
        income=float(income_t),
        nn=int(gcfg.n * gcfg.n),
        survprob=np.asarray(survprob),
        gret_sh=np.asarray(gret_sh),
        r=float(fp.r),
        v_next=np.asarray(v_next_pay),
        gcash_grid=np.asarray(gcash),
        ghouse_grid=np.asarray(ghouse),
        interp_method=args.interp_method,
    )

    aux_nopay = AuxVParams(**{**asdict(aux_pay), "otcost": 0.0, "v_next": np.asarray(v_next_nopay)})

    cases = [
        ("pay_buy_part", aux_pay, model_pay, "buy", True, lambda c, h, ppc, otc: h * (1 - fp.adjcost - args.ppt) + c - otc - ppc),
        ("pay_zero_part", aux_pay, model_pay, "zero", True, lambda c, h, ppc, otc: h * (1 - fp.adjcost - args.ppt) + c - otc - ppc),
        ("pay_buy_nopart", aux_pay, model_pay, "buy", False, lambda c, h, ppc, otc: h * (1 - fp.adjcost - args.ppt) + c),
        ("pay_zero_nopart", aux_pay, model_pay, "zero", False, lambda c, h, ppc, otc: h * (1 - fp.adjcost - args.ppt) + c),
        ("pay_keep_part", aux_pay, model_pay, "keep", True, lambda c, h, ppc, otc: h * (-args.ppt) + c - otc - ppc),
        ("pay_keep_nopart", aux_pay, model_pay, "keep", False, lambda c, h, ppc, otc: h * (-args.ppt) + c),
        ("nopay_buy_nopart", aux_nopay, model_nopay, "buy", False, lambda c, h, ppc, otc: h * (1 - fp.adjcost - args.ppt) + c),
        ("nopay_zero_nopart", aux_nopay, model_nopay, "zero", False, lambda c, h, ppc, otc: h * (1 - fp.adjcost - args.ppt) + c),
        ("nopay_keep_nopart", aux_nopay, model_nopay, "keep", False, lambda c, h, ppc, otc: h * (-args.ppt) + c),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str | int]] = []
    best_case = None
    best_v = -np.inf

    for name, aux, model_fn_np, h_mode, can_participate, budget_fn in cases:
        b = float(budget_fn(thecash, thehouse, ppc_t, otc_t))
        c, a, h, v_case = _solve_one_state_gpu_continuous(
            thecash=thecash,
            thehouse=thehouse,
            aux_params=aux,
            b=b,
            h_mode=h_mode,
            can_participate=can_participate,
            fp=fp,
            minhouse2=minhouse2_t,
            model_fn_np=model_fn_np,
            maxiter=int(args.maxiter),
            step_size=float(args.step_size),
        )
        dec = _decompose_one(np.array([c, a, h], dtype=float), aux, thecash, thehouse, model_fn_np)

        summary_rows.append({
            "case": name,
            "budget_b": b,
            "c": c,
            "a": a,
            "h": h,
            "u_now": dec["u_now"],
            "ev_term_raw": dec["ev_term_raw"],
            "ev_term": dec["ev_term"],
            "total_v_decomp": dec["total_v"],
            "total_v_solver": v_case,
        })

        detail_csv = out_dir / f"{name}_details.csv"
        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shock_idx", "weight", "housing_nn", "cash_nn", "interp_v_next", "interp_pow_1_minus_rho", "weighted_contrib"])
            for k in range(len(dec["weights"])):
                w.writerow([
                    k,
                    float(dec["weights"][k]),
                    float(dec["housing_nn"][k]),
                    float(dec["cash_nn"][k]),
                    float(dec["interp_v_next"][k]),
                    float(dec["interp_pow_1_minus_rho"][k]),
                    float(dec["weighted_contrib"][k]),
                ])

        if dec["total_v"] > best_v:
            best_v = float(dec["total_v"])
            best_case = name

    summary_rows_sorted = sorted(summary_rows, key=lambda r: float(r["total_v_decomp"]), reverse=True)
    summary_csv = out_dir / "candidate_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(summary_rows_sorted)

    print("\n===== T-1 V1 candidate decomposition =====")
    print(f"state: cash_idx={i}, house_idx={j}, thecash={thecash:.6f}, thehouse={thehouse:.6f}")
    print(f"t_index={t} (T-1), interp={args.interp_method}, gyp={gyp_t:.6f}, income={income_t:.6f}")
    print(f"best_case={best_case}, best_total_v={best_v:.6f}")
    print(f"[save] summary -> {summary_csv}")
    print(f"[save] details -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
