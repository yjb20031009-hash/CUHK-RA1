"""MATLAB-like interp2 helpers in JAX (regular grid)."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def _as_array(x) -> Array:
    return jnp.asarray(x)


def _clamp(q: Array, grid: Array) -> Array:
    return jnp.clip(q, grid[0], grid[-1])


def _searchsorted_clipped(grid: Array, q: Array) -> Array:
    """Return i such that grid[i] <= q < grid[i+1], clipped to valid range."""
    i = jnp.searchsorted(grid, q, side="right") - 1
    return jnp.clip(i, 0, grid.shape[0] - 2)


def _validate_inputs(x: Array, y: Array, v: Array) -> None:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if v.ndim != 2 or v.shape != (y.size, x.size):
        raise ValueError("v must have shape (len(y), len(x))")


def interp2_bilinear(x_grid, y_grid, v, xq, yq) -> Array:
    """Bilinear interpolation on regular grid (with clamped query points)."""
    x_grid = _as_array(x_grid)
    y_grid = _as_array(y_grid)
    v = _as_array(v)
    xq = _as_array(xq)
    yq = _as_array(yq)
    xq, yq = jnp.broadcast_arrays(xq, yq)

    _validate_inputs(x_grid, y_grid, v)

    xq = _clamp(xq, x_grid)
    yq = _clamp(yq, y_grid)

    ix = _searchsorted_clipped(x_grid, xq)
    iy = _searchsorted_clipped(y_grid, yq)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]

    tx = jnp.where(x1 > x0, (xq - x0) / (x1 - x0), 0.0)
    ty = jnp.where(y1 > y0, (yq - y0) / (y1 - y0), 0.0)

    v00 = v[iy, ix]
    v10 = v[iy, ix + 1]
    v01 = v[iy + 1, ix]
    v11 = v[iy + 1, ix + 1]

    v0 = v00 * (1.0 - tx) + v10 * tx
    v1 = v01 * (1.0 - tx) + v11 * tx
    return v0 * (1.0 - ty) + v1 * ty


def interp2_nearest(x_grid, y_grid, v, xq, yq) -> Array:
    """Nearest-neighbor interpolation on regular grid (with clamped query points)."""
    x_grid = _as_array(x_grid)
    y_grid = _as_array(y_grid)
    v = _as_array(v)
    xq = _as_array(xq)
    yq = _as_array(yq)
    xq, yq = jnp.broadcast_arrays(xq, yq)

    _validate_inputs(x_grid, y_grid, v)

    xq = _clamp(xq, x_grid)
    yq = _clamp(yq, y_grid)

    ix_r = jnp.clip(jnp.searchsorted(x_grid, xq, side="left"), 0, x_grid.shape[0] - 1)
    iy_r = jnp.clip(jnp.searchsorted(y_grid, yq, side="left"), 0, y_grid.shape[0] - 1)
    ix_l = jnp.maximum(ix_r - 1, 0)
    iy_l = jnp.maximum(iy_r - 1, 0)

    ix = jnp.where(jnp.abs(xq - x_grid[ix_l]) <= jnp.abs(xq - x_grid[ix_r]), ix_l, ix_r)
    iy = jnp.where(jnp.abs(yq - y_grid[iy_l]) <= jnp.abs(yq - y_grid[iy_r]), iy_l, iy_r)
    return v[iy, ix]


def interp2_regular(
    x,
    y,
    v,
    xq,
    yq,
    method: str = "linear",
    *,
    extrapval=jnp.nan,
    bounds: str = "nan",
) -> Array:
    """MATLAB-like wrapper around regular-grid interpolation.

    bounds:
      - "nan": out-of-bound values replaced by extrapval
      - "clip": out-of-bound values are clamped to grid boundary before interpolation
    """
    x = _as_array(x)
    y = _as_array(y)
    v = _as_array(v)
    xq = _as_array(xq)
    yq = _as_array(yq)
    xq, yq = jnp.broadcast_arrays(xq, yq)

    _validate_inputs(x, y, v)

    outside = (xq < x[0]) | (xq > x[-1]) | (yq < y[0]) | (yq > y[-1])

    if method.lower() == "nearest":
        out = interp2_nearest(x, y, v, xq, yq)
    elif method.lower() == "linear":
        out = interp2_bilinear(x, y, v, xq, yq)
    else:
        raise ValueError("method must be 'linear' or 'nearest'")

    if bounds.lower() == "clip":
        return out
    if bounds.lower() == "nan":
        return jnp.where(outside, extrapval, out)
    raise ValueError("bounds must be 'nan' or 'clip'")
