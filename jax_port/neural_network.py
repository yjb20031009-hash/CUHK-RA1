"""JAX rewrite for the workflow in `Neural_Network.m`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers, stax


Array = jnp.ndarray


def build_training_data(
    gcash: Array,
    ghouse: Array,
    tn: int,
    a: Array,
    h: Array,
    c: Array,
    a1: Array,
    h1: Array,
    c1: Array,
) -> tuple[Array, Array]:
    """Replicates MATLAB meshgrid+reshape preprocessing."""
    gcash_mesh, ghouse_mesh, t_mesh = jnp.meshgrid(
        gcash,
        ghouse,
        jnp.arange(1, tn + 1),
        indexing="ij",
    )
    x = jnp.stack(
        [gcash_mesh.reshape(-1), ghouse_mesh.reshape(-1), t_mesh.reshape(-1)], axis=1
    )
    y = jnp.stack(
        [
            a.reshape(-1),
            h.reshape(-1),
            c.reshape(-1),
            a1.reshape(-1),
            h1.reshape(-1),
            c1.reshape(-1),
        ],
        axis=1,
    )
    return x, y


def init_mlp(rng_key: jax.Array, hidden_width: int = 100):
    init_fn, predict_fn = stax.serial(
        stax.Dense(hidden_width),
        stax.Relu,
        stax.Dense(6),
    )
    _, params = init_fn(rng_key, (-1, 3))
    return params, predict_fn


def train_mlp(
    x: Array,
    y: Array,
    params,
    predict_fn,
    *,
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
):
    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(params)

    @jax.jit
    def loss_fn(model_params, xb, yb):
        pred = predict_fn(model_params, xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def step(step_idx, state, xb, yb):
        model_params = get_params(state)
        grads = jax.grad(loss_fn)(model_params, xb, yb)
        return opt_update(step_idx, grads, state)

    n = x.shape[0]
    steps = 0
    key = jax.random.key(seed)

    for _ in range(epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)
        x_epoch = x[perm]
        y_epoch = y[perm]
        for i in range(0, n, batch_size):
            xb = x_epoch[i : i + batch_size]
            yb = y_epoch[i : i + batch_size]
            opt_state = step(steps, opt_state, xb, yb)
            steps += 1

    return get_params(opt_state)


def predict_mlp(x: Array, params, predict_fn) -> Array:
    return predict_fn(params, x)
