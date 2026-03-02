# 调用关系梳理（MATLAB ↔ JAX）

## 1) 本次转写对应关系

- `tauchenHussey.m` → `jax_port/tauchen_hussey.py`
- `Neural_Network.m` → `jax_port/neural_network.py`
- MATLAB `interp2`（规则网格常用调用）→ `jax_port/interp2.py`
- MATLAB `fmincon`（约束优化调用）→ `jax_port/fmincon.py`
- `jax_port/__init__.py` 作为统一导出入口

---

## 2) MATLAB 侧调用关系

### 2.1 `tauchenHussey.m`（纯函数）

```text
tauchenHussey
├─ gaussnorm
│  └─ gausshermite
└─ norm_pdf
```

### 2.2 `Neural_Network.m`（脚本流程）

```text
Neural_Network.m
├─ meshgrid + reshape（构造 inputData / outputData）
├─ trainNetwork（MATLAB DL Toolbox）
└─ predict
```

### 2.3 内置函数连接点（关键）

```text
my_estimation_prepost*.m / pictures.m / decomposition_*.m
└─ interp2(...)

mymain_se.m / picture_inner_func.m / test.m
└─ fmincon(...)
```

---

## 3) JAX 侧调用关系

### 3.1 `jax_port/tauchen_hussey.py`

```text
tauchen_hussey
├─ _gaussnorm
│  └─ _gausshermite
└─ _norm_pdf
```

### 3.2 `jax_port/neural_network.py`

```text
build_training_data
init_mlp
train_mlp
├─ loss_fn (jit)
└─ step (jit + grad)
predict_mlp
```

### 3.3 `jax_port/interp2.py`

```text
interp2_regular
├─ _find_cell_indices (linear)
└─ nearest / linear 分支
```

### 3.4 `jax_port/fmincon.py`

```text
fmincon
├─ _wrap_fun
├─ _wrap_jac (jax.grad)
└─ scipy.optimize.minimize (SLSQP / trust-constr)
```

### 3.5 `jax_port/my_auxv_cal.py`

```text
my_auxv_cal
├─ jax.lax.cond (四分支现金流)
└─ model_fn(housing_nn, cash_nn)
```

### 3.6 `jax_port/__init__.py` 导出关系

```text
__init__.py
├─ tauchen_hussey
├─ build_training_data / init_mlp / train_mlp / predict_mlp
├─ interp2_regular
├─ fmincon / FminconResult
└─ my_auxv_cal / AuxVParams
```
