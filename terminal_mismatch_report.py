#!/usr/bin/env python3
"""Terminal-period mismatch report for MATLAB vs Python value-function outputs.

Example:
  python terminal_mismatch_report.py \
    --python-mat python_quick_test_result.mat \
    --matlab-mat matlab_quick_test_result.mat \
    --out-csv terminal_value_comparison.csv \
    --heatmap-dir terminal_value_heatmaps
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat


def _squeeze_3d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D after squeeze, got shape={arr.shape}")
    return arr.astype(float, copy=False)


def _load_first(data: dict, keys: Iterable[str], *, side: str) -> np.ndarray:
    for k in keys:
        if k in data:
            return np.asarray(data[k])
    raise KeyError(f"Could not find keys {list(keys)} in {side} mat file")


def _load_value_pair(py_data: dict, mat_data: dict, which_value: str) -> tuple[np.ndarray, np.ndarray]:
    w = which_value.upper()
    if w not in {"V", "V1"}:
        raise ValueError("which_value must be one of {'V', 'V1'}")

    py_keys = [f"{w}_py", w]
    mat_keys = [w, f"{w}_mat", f"{w}_m"]
    py_arr = _squeeze_3d(_load_first(py_data, py_keys, side="python"), f"python:{w}")
    mat_arr = _squeeze_3d(_load_first(mat_data, mat_keys, side="matlab"), f"matlab:{w}")
    if py_arr.shape != mat_arr.shape:
        raise ValueError(f"shape mismatch for {w}: python={py_arr.shape}, matlab={mat_arr.shape}")
    return py_arr, mat_arr


def _period_metrics(py_t: np.ndarray, mat_t: np.ndarray, eps: float) -> dict[str, float]:
    diff = py_t - mat_t
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.abs(mat_t), eps)
    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "median_abs_diff": float(np.median(abs_diff)),
        "p95_abs_diff": float(np.percentile(abs_diff, 95)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
    }


def _save_heatmap(arr: np.ndarray, title: str, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] skip heatmap {out_png.name}: matplotlib unavailable ({exc})")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)
    im = ax.imshow(arr, cmap="magma", aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("house index")
    ax.set_ylabel("cash index")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("abs diff")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _pick_existing(path_candidates: Sequence[str]) -> str | None:
    for p in path_candidates:
        if Path(p).exists():
            return p
    return None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate terminal backward mismatch summary for MATLAB vs Python .mat outputs")
    parser.add_argument("--python-mat", default=None, help="Python .mat file (e.g. python_quick_test_result.mat)")
    parser.add_argument("--matlab-mat", default=None, help="MATLAB .mat file")
    parser.add_argument("--which-values", nargs="+", default=["V", "V1"], help="Any subset of: V V1")
    parser.add_argument("--last-k", type=int, default=10, help="How many terminal periods to print/save (T ... T-k+1)")
    parser.add_argument("--eps", type=float, default=1e-8, help="Denominator floor for relative error")
    parser.add_argument("--out-csv", default="terminal_value_comparison.csv", help="Output CSV path")
    parser.add_argument("--heatmap-dir", default="terminal_value_heatmaps", help="Output directory for heatmaps")
    parser.add_argument("--topk-states", type=int, default=20, help="Export top-k largest abs-diff states per period (0 to disable)")
    parser.add_argument("--topk-dir", default="terminal_topk_states", help="Output directory for top-k per-period CSVs")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable heatmap generation")
    args, _ = parser.parse_known_args(argv)

    if args.python_mat is None:
        args.python_mat = _pick_existing((
            "python_quick_test_result.mat",
            "fresh_jax_value_policy_gpu2.mat",
            "fresh_jax_value_policy_gpu.mat",
            "fresh_jax_value_policy_continuous.mat",
        ))
    if args.matlab_mat is None:
        args.matlab_mat = _pick_existing((
            "matlab_quick_test_result.mat",
            "matlab_value_policy.mat",
            "mat_value_policy.mat",
        ))
    if args.python_mat is None or args.matlab_mat is None:
        raise SystemExit(
            "Missing input .mat files. Provide --python-mat and --matlab-mat, "
            "or place default files in current directory."
        )

    py_data = loadmat(args.python_mat)
    mat_data = loadmat(args.matlab_mat)

    rows: list[dict[str, float | int | str]] = []

    print("\n===== Terminal-backward comparison summary =====")
    header = (
        "which_value", "time_label", "t_index_python", "max_abs_diff", "mean_abs_diff",
        "median_abs_diff", "p95_abs_diff", "max_rel_diff", "mean_rel_diff"
    )
    print(" ".join(h.rjust(14) for h in header))

    for which in args.which_values:
        py_v, mat_v = _load_value_pair(py_data, mat_data, which)
        tn = py_v.shape[2]
        k = int(max(1, min(args.last_k, tn)))

        for offs in range(k):
            t = tn - 1 - offs
            py_t = py_v[:, :, t]
            mat_t = mat_v[:, :, t]
            met = _period_metrics(py_t, mat_t, eps=float(args.eps))
            label = "T" if offs == 0 else f"T-{offs}"
            row = {
                "which_value": which.upper(),
                "time_label": label,
                "t_index_python": int(t),
                **met,
            }
            rows.append(row)
            print(
                f"{row['which_value']:>14} {row['time_label']:>14} {row['t_index_python']:>14d} "
                f"{row['max_abs_diff']:>14.6f} {row['mean_abs_diff']:>14.6f} {row['median_abs_diff']:>14.6f} "
                f"{row['p95_abs_diff']:>14.6f} {row['max_rel_diff']:>14.6f} {row['mean_rel_diff']:>14.6f}"
            )

            if not args.no_heatmap:
                abs_diff = np.abs(py_t - mat_t)
                out_png = Path(args.heatmap_dir) / f"{which.upper()}_{label.replace('-', 'm')}_abs_diff.png"
                _save_heatmap(abs_diff, f"{which.upper()} {label} abs diff", out_png)

            if int(args.topk_states) > 0:
                abs_diff = np.abs(py_t - mat_t)
                rel_diff = abs_diff / np.maximum(np.abs(mat_t), float(args.eps))
                flat_idx = np.argsort(abs_diff.ravel())[::-1][: int(args.topk_states)]
                cash_idx, house_idx = np.unravel_index(flat_idx, abs_diff.shape)
                out_topk = Path(args.topk_dir) / f"{which.upper()}_{label.replace('-', 'm')}_topk.csv"
                out_topk.parent.mkdir(parents=True, exist_ok=True)
                with out_topk.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["rank", "cash_idx", "house_idx", "py_value", "mat_value", "abs_diff", "rel_diff"])
                    for rank, ci, hi in zip(range(1, len(flat_idx) + 1), cash_idx, house_idx):
                        w.writerow([
                            rank,
                            int(ci),
                            int(hi),
                            float(py_t[ci, hi]),
                            float(mat_t[ci, hi]),
                            float(abs_diff[ci, hi]),
                            float(rel_diff[ci, hi]),
                        ])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[save] summary csv -> {out_csv}")
    if not args.no_heatmap:
        print(f"[save] heatmaps    -> {Path(args.heatmap_dir).resolve()}")
    if int(args.topk_states) > 0:
        print(f"[save] top-k csvs  -> {Path(args.topk_dir).resolve()}")


if __name__ == "__main__":
    main()
