"""JAX implementation of MATLAB `tauchenHussey.m`."""

from __future__ import annotations

import math

import jax.numpy as jnp


def _norm_pdf(x: jnp.ndarray, mu: jnp.ndarray | float, s2: float) -> jnp.ndarray:
    return 1.0 / jnp.sqrt(2.0 * jnp.pi * s2) * jnp.exp(-((x - mu) ** 2) / (2.0 * s2))


def _gausshermite(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Hermite nodes and weights, matching MATLAB implementation."""
    maxit = 10
    eps = 3e-14
    pim4 = 0.7511255444649425

    x = [0.0] * n
    w = [0.0] * n

    m = (n + 1) // 2
    z = 0.0
    for i in range(1, m + 1):
        if i == 1:
            z = math.sqrt((2 * n + 1) - 1.85575 * (2 * n + 1) ** (-0.16667))
        elif i == 2:
            z = z - 1.14 * (n**0.426) / z
        elif i == 3:
            z = 1.86 * z - 0.86 * x[0]
        elif i == 4:
            z = 1.91 * z - 0.91 * x[1]
        else:
            z = 2.0 * z - x[i - 3]

        for _ in range(maxit):
            p1 = pim4
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = z * math.sqrt(2.0 / j) * p2 - math.sqrt((j - 1.0) / j) * p3
            pp = math.sqrt(2.0 * n) * p2
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= eps:
                break

        x[i - 1] = z
        x[n - i] = -z
        wi = 2.0 / (pp * pp)
        w[i - 1] = wi
        w[n - i] = wi

    x = jnp.array(x[::-1])
    w = jnp.array(w)
    return x, w


def _gaussnorm(n: int, mu: float, s2: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    x0, w0 = _gausshermite(n)
    x = x0 * jnp.sqrt(2.0 * s2) + mu
    w = w0 / jnp.sqrt(jnp.pi)
    return x, w


def tauchen_hussey(
    n: int,
    mu: float,
    rho: float,
    sigma: float,
    base_sigma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate AR(1) process using Tauchen-Hussey quadrature."""
    z, w = _gaussnorm(n, mu, base_sigma**2)
    ez_prime = (1.0 - rho) * mu + rho * z[:, None]
    numer = w[None, :] * _norm_pdf(z[None, :], ez_prime, sigma**2)
    denom = _norm_pdf(z[None, :], mu, base_sigma**2)
    zprob = numer / denom
    zprob = zprob / jnp.sum(zprob, axis=1, keepdims=True)
    return z, zprob
