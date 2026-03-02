"""MATLAB-like interp2 helpers in JAX.

Implements regular-grid 2D interpolation for the common call pattern:
    interp2(x, y, V, xq, yq, method)
where x corresponds to columns of V and y to rows of V.
"""

from __future__ import annotations

import jax.numpy as jnp


Array = jnp.ndarray


def _as_array(x) -> Array:
    return jnp.asarray(x)


def _find_cell_indices(grid: Array, q: Array) -> tuple[Array, Array, Array]:
    """Return left/right indices and normalized local coordinate in [0,1]."""
    idx_right = jnp.searchsorted(grid, q, side="right")
    idx_right = jnp.clip(idx_right, 1, grid.size - 1)
    idx_left = idx_right - 1

    g0 = grid[idx_left]
    g1 = grid[idx_right]
    denom = jnp.maximum(g1 - g0, 1e-15)
    t = (q - g0) / denom
    t = jnp.clip(t, 0.0, 1.0)
    return idx_left, idx_right, t


def interp2_regular(
    x,
    y,
    v,
    xq,
    yq,
    method: str = "linear",
    *,
    extrapval=jnp.nan,
) -> Array:
    """Regular-grid interpolation similar to MATLAB ``interp2``.

    Args:
        x: 1D increasing grid for columns of ``v`` (size nx)
        y: 1D increasing grid for rows of ``v`` (size ny)
        v: 2D values, shape (ny, nx)
        xq, yq: query locations (broadcastable same shape)
        method: "linear" or "nearest"
        extrapval: fill value for out-of-domain points
    """
    x = _as_array(x)
    y = _as_array(y)
    v = _as_array(v)
    xq = _as_array(xq)
    yq = _as_array(yq)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if v.ndim != 2 or v.shape != (y.size, x.size):
        raise ValueError("v must have shape (len(y), len(x))")

    xq, yq = jnp.broadcast_arrays(xq, yq)

    outside = (xq < x[0]) | (xq > x[-1]) | (yq < y[0]) | (yq > y[-1])

    if method.lower() == "nearest":
        ix_r = jnp.clip(jnp.searchsorted(x, xq, side="left"), 0, x.size - 1)
        iy_r = jnp.clip(jnp.searchsorted(y, yq, side="left"), 0, y.size - 1)
        ix_l = jnp.maximum(ix_r - 1, 0)
        iy_l = jnp.maximum(iy_r - 1, 0)

        choose_left_x = jnp.abs(xq - x[ix_l]) <= jnp.abs(xq - x[ix_r])
        choose_left_y = jnp.abs(yq - y[iy_l]) <= jnp.abs(yq - y[iy_r])
        ix = jnp.where(choose_left_x, ix_l, ix_r)
        iy = jnp.where(choose_left_y, iy_l, iy_r)

        out = v[iy, ix]
        return jnp.where(outside, extrapval, out)

    if method.lower() != "linear":
        raise ValueError("method must be 'linear' or 'nearest'")

    ix0, ix1, tx = _find_cell_indices(x, xq)
    iy0, iy1, ty = _find_cell_indices(y, yq)

    v00 = v[iy0, ix0]
    v10 = v[iy0, ix1]
    v01 = v[iy1, ix0]
    v11 = v[iy1, ix1]

    out = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    return jnp.where(outside, extrapval, out)
