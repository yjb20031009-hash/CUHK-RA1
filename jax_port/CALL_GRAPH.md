# 调用关系梳理（MATLAB ↔ JAX）

## 1) 本次转写对应关系

- `tauchenHussey.m` → `jax_port/tauchen_hussey.py`
- `Neural_Network.m` → `jax_port/neural_network.py`
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

说明：
- `tauchenHussey` 先调用 `gaussnorm` 生成离散节点和权重，再用 `norm_pdf` 构造转移概率矩阵，并逐行归一化。
- `gaussnorm` 内部调用 `gausshermite` 获取高斯-厄米求积节点/权重。

### 2.2 `Neural_Network.m`（脚本流程）

```text
Neural_Network.m
├─ meshgrid + reshape（构造 inputData / outputData）
├─ trainNetwork（MATLAB DL Toolbox）
└─ predict
```

说明：
- 该文件不是函数而是脚本，依赖外部工作区变量（如 `gcash/ghouse/tn/A/H/C/A1/H1/C1`）。
- 核心逻辑是“数据展平 → MLP 训练 → 预测后 reshape 可视化”。

### 2.3 与主工程的连接点（关键）

```text
my_solution.m / my_estimation_*.m / picture_inner_func.m / mymain_se.m
└─ tauchenHussey(...)
```

说明：
- `tauchenHussey.m` 在多个估计/仿真脚本中被重复调用，是核心公共数值模块。
- `Neural_Network.m` 目前与主估计链路耦合较弱，主要用于拟合/可视化实验。

---

## 3) JAX 侧调用关系

### 3.1 `jax_port/tauchen_hussey.py`

```text
tauchen_hussey
├─ _gaussnorm
│  └─ _gausshermite
└─ _norm_pdf
```

对应 MATLAB：
- `tauchenHussey` ↔ `tauchen_hussey`
- `gaussnorm` ↔ `_gaussnorm`
- `gausshermite` ↔ `_gausshermite`
- `norm_pdf` ↔ `_norm_pdf`

### 3.2 `jax_port/neural_network.py`

```text
build_training_data  # 对应 meshgrid + reshape
init_mlp             # 定义 MLP 结构
train_mlp
├─ loss_fn (jit)
└─ step (jit + grad)
predict_mlp
```

说明：
- `build_training_data` 对齐 MATLAB 脚本中的输入输出展平过程。
- `init_mlp/train_mlp/predict_mlp` 对齐 MATLAB 的 `trainNetwork/predict` 工作流。

### 3.3 `jax_port/__init__.py` 导出关系

```text
__init__.py
├─ from .tauchen_hussey import tauchen_hussey
└─ from .neural_network import build_training_data, init_mlp, train_mlp, predict_mlp
```

---

## 4) 建议的迁移顺序（按依赖）

1. 先将所有调用 `tauchenHussey` 的 MATLAB 文件改为调用 `jax_port.tauchen_hussey`（低风险）。
2. 再把 `Neural_Network.m` 上游产生 `A/H/C/A1/H1/C1` 的模块函数化，接入 `build_training_data`。
3. 最后迁移 `my_estimation_prepostdid1*.m` 与 `my_solution.m` 的优化主循环（`cmaes2` 链路）。
