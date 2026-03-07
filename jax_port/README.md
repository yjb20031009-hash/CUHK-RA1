# MATLAB 到 JAX 转写

本目录提供了对仓库中关键 MATLAB 代码/内置调用的 JAX(Python) 实现：

- `tauchenHussey.m` → `tauchen_hussey.py`
- `Neural_Network.m` → `neural_network.py`
- MATLAB `interp2`（规则网格常用形式）→ `interp2.py`
- MATLAB `fmincon`（约束优化接口近似）→ `fmincon.py`
- MATLAB `cmaes2.m`（核心 CMA-ES）→ `cmaes2_jax.py`
- MATLAB `my_auxV_cal.m`（底层目标函数）→ `my_auxv_cal.py`
- MATLAB `mymain_se.m`（policy function 主求解）→ `mymain_se.py`（离散搜索近似）
- MATLAB `my_estimation_prepost.m` → `my_estimation_prepost.py`（工程化结构转写）
- MATLAB `my_estimation_prepostdid1.m` → `my_estimation_prepostdid1.py`
- `my_estimation_prepostdid1_high.m` → `my_estimation_prepostdid1_high.py`
- `my_estimation_prepostdid1_low.m` → `my_estimation_prepostdid1_low.py`

> 说明：像 `my_solution.m` / `my_estimation_*.m` 这类大规模估计脚本依赖大量全局变量和外部函数，建议下一步按模块（状态转移、价值函数、目标函数）继续拆分转写。

## 快速示例

```python
import jax
import jax.numpy as jnp
from jax_port import (
    tauchen_hussey,
    build_training_data,
    init_mlp,
    train_mlp,
    predict_mlp,
    interp2_regular,
    fmincon,
)

# Tauchen-Hussey
z, zprob = tauchen_hussey(n=7, mu=0.0, rho=0.9, sigma=0.2, base_sigma=0.2)

# interp2 (regular grid)
x = jnp.array([0.0, 1.0, 2.0])
y = jnp.array([0.0, 1.0])
V = jnp.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
val = interp2_regular(x, y, V, xq=0.5, yq=0.5, method="linear")

# fmincon-like optimization
fun = lambda u: (u[0]-1.0) ** 2 + (u[1]+2.0) ** 2
res = fmincon(fun, x0=[0.0, 0.0], lb=[-1.0, -3.0], ub=[2.0, 1.0])
```


`my_auxv_cal.py` 提供了 `AuxVParams` 数据结构和 `my_auxv_cal` 函数，用于复现四种分支现金流与下一期价值聚合逻辑。


> 运行 DID1 封装/`run_my_solution` 时，除 `mySample_pre10.mat` 外，还需要对应的 moments 文件：`Sample_did_nosample.mat`、`Sample_did_nosample_high.mat`、`Sample_did_nosample_low.mat`（用于 `W` 和 `beta1..beta6`）。
