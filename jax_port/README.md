# MATLAB 到 JAX 转写
本目录提供了对仓库中关键 MATLAB 代码/内置调用的 JAX(Python) 实现：
原代码架构：
First:
最外层是My solution： 
1.my_estimation_prepostdid1_high
2.cmaes2('my_estimation_prepostdid1', ....
3.cmaes2( 'my_estimation_prepostdid1_low' ,....;
4.cmaes2('my_estimation_prepostdid1_high',....;

中间cmaes2 -> 一个封装好的直接使用的函数
my_estimation_prepostdid1（这3个区别不大） -> 最重要的外层函数 —>调用的mymain_se作Policy function， 还有tauchenHussey函数
mymain_se -> policy function ->输入一组可变的参数，通过grid search和fmincon结合的方法，求解policy function（打格子的方法是先取ln，然后对ln(cash)均匀等分，再求exp(lncash)转换回cash） ->要调用my_auxV_cal，还要使用内置函数fmicon。my_long_loop说是写了要用，但是原代码没看见在哪

最底层-my_auxV_cal 目标函数
	tauchenHussey 离散函数
	Neutral network 神经网络训练+画图，没搞懂干嘛用的，主函数没有调用
    重写了matlab里的内置函数intercp2以及fmicon，因为Jax没有



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
