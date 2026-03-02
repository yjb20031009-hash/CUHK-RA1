# MATLAB 到 JAX 转写

本目录提供了对仓库中关键 MATLAB 代码的 JAX 实现：

- `tauchenHussey.m` → `tauchen_hussey.py`
- `Neural_Network.m` → `neural_network.py`

> 说明：像 `my_solution.m` / `my_estimation_*.m` 这类大规模估计脚本依赖大量全局变量和外部函数，建议下一步按模块（状态转移、价值函数、目标函数）继续拆分转写。

## 快速示例

```python
import jax
import jax.numpy as jnp
from jax_port import tauchen_hussey, build_training_data, init_mlp, train_mlp, predict_mlp

# Tauchen-Hussey
z, zprob = tauchen_hussey(n=7, mu=0.0, rho=0.9, sigma=0.2, base_sigma=0.2)

# Neural network
gcash = jnp.linspace(0.0, 1.0, 10)
ghouse = jnp.linspace(0.0, 1.0, 8)
tn = 4
shape = (gcash.size, ghouse.size, tn)
zeros = jnp.zeros(shape)

x, y = build_training_data(gcash, ghouse, tn, zeros, zeros, zeros, zeros, zeros, zeros)
params, predict_fn = init_mlp(jax.random.key(0))
trained = train_mlp(x, y, params, predict_fn, epochs=5)
yhat = predict_mlp(x, trained, predict_fn)
```
