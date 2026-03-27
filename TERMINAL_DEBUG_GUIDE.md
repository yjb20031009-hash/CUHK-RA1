# Terminal 期对不齐排查指南（MATLAB vs Python）

这个指南对应你现在看到的现象：
- `T`/`T-1` 的 `max_abs_diff` 很大，之后快速衰减。

## 1) 先判断是“值函数定义差异”还是“策略选择差异”

对每个状态 `(cash_i, house_j)` 在同一个 `t`，做 3 个量：

1. `V_py_stored`: Python 保存的 `V[i,j,t]`
2. `V_py_eval`: 用 **Python 当前的 Bellman 目标函数**，在 `(C_py,A_py,H_py)` 上重新评估得到的值
3. `V_mat_eval`: 用 **同一个 Python Bellman 目标函数**，在 `(C_mat,A_mat,H_mat)` 上评估得到的值

解释：
- 如果 `|V_py_stored - V_py_eval|` 大：是 Python 端“求解器一致性/收敛”问题。
- 如果 `|V_mat - V_mat_eval|` 大：是 MATLAB 导出或口径不一致问题。
- 如果两边 residual 都小，但 `|V_py_eval - V_mat_eval|` 大：是策略选点不同（优化路径问题）。

## 2) 先只盯 `t = T`（你图里 `t_index_python=40`）

从你的截图看，`T` 期最大绝对差在 8k 量级，优先检查终值输入：
- `V_next` 是否完全一致（尤其是 `V1[:,:,T] = V[:,:,T]` 的口径）。
- 折现/生存概率/收益率是否逐项一致。
- 插值是否都走 `spline`，且边界处理一致（clip vs extrapolate）。

## 3) 对“差异最大”的前 20 个状态做 case-level 分解

在每个大差异状态，打印 9 个候选 case（pay / nopay）各自目标值，并记录：
- best case id
- second-best case id
- best-second gap

解释：
- 如果 best-second gap 很小，说明是“近似并列最优”，不同求解器/初值会选不同 case（正常会扩散到前几期）。
- 如果 gap 很大但选错，说明约束、边界或 objective 实现有 bug。

## 4) 固定一边策略，单边回代验证

做两个反事实：
- 用 MATLAB 策略 `(C_mat,A_mat,H_mat)`，放入 Python Bellman evaluator，生成 `V_mat_on_py_obj`。
- 用 Python 策略 `(C_py,A_py,H_py)`，放入同一 evaluator，生成 `V_py_on_py_obj`。

这样可以把“优化器误差”和“目标函数实现差异”拆开。

## 5) 先把可变因素锁死（用于定位，不是最终配置）

建议在排查阶段固定：
- `interp_method='spline'`
- `gpu_use_warmstart=False`
- `gpu_add_boundary_candidates=False`

这样可减少路径依赖，让你更容易复现并定位 `T` 期差异来源。

## 6) 快速优先级（按收益排序）

1. `T` 期 top-20 差异状态的 case ranking 是否一致
2. 同状态下预算约束是否一致（尤其是 `h=0`、`h=minhouse2`、`A≈0.25` 边界）
3. 插值输入是否触边（`cash_nn/housing_nn` 被 clip 的比例）
4. 如果以上都一致，再看优化器容差、multi-start 初值和步长策略

## 7) 你这张图的直观判断

- 差异从 `T` 到 `T-9` 快速衰减，通常不是“全局公式错”，更像“末期少数状态选点差异 + 递推传播”。
- 优先做第 3 步（case-level 分解）和第 4 步（固定策略回代），最快能锁定是“目标函数”还是“策略优化”层面的问题。

## 8) 一键脚本（直接运行）

仓库里提供了 `terminal_mismatch_report.py`，可直接生成你截图那种终值回溯表：

```bash
python terminal_mismatch_report.py \
  --python-mat python_quick_test_result.mat \
  --matlab-mat matlab_quick_test_result.mat \
  --which-values V V1 \
  --last-k 10 \
  --topk-states 20 \
  --out-csv terminal_value_comparison.csv \
  --heatmap-dir terminal_value_heatmaps
```

输出：
- `terminal_value_comparison.csv`
- `terminal_value_heatmaps/`（每个时期一张 abs-diff 热图）
- `terminal_topk_states/`（每个时期 top-k 最大误差状态，含 cash/house 索引）

如果你在 Jupyter 里直接运行，也可以不带参数（脚本会自动找常见文件名）：

```bash
python terminal_mismatch_report.py
```

## 9) 拆解 T-1 的 V1 各 candidate（u_now / transition / continuation）

如果你要看单个状态在 `T-1` 的 9 个 candidate 的细项分解，运行：

```bash
python decompose_tminus1_v1_candidates.py \
  --policy-mat fresh_jax_value_policy_gpu2.mat \
  --cash-idx 20 \
  --house-idx 10 \
  --ppt 0.04 \
  --ppcost 0.024689 \
  --otcost 0.097854 \
  --rho 9.759 \
  --delta 0.9871 \
  --psi 0.67324 \
  --interp-method spline
```

输出：
- `candidate_summary.csv`：每个 case 的 `u_now / ev_term_raw / ev_term(floor后) / total_v / solver_v / c,a,h`
- `*_details.csv`：每个 shock 下的 `cash_nn / housing_nn / interp_v_next / interp_pow_1_minus_rho / weighted_contrib / weight`

在 Jupyter 里你也可以直接先跑：

```bash
python decompose_tminus1_v1_candidates.py
```

脚本会自动尝试默认 `.mat` 文件名，并默认取右上角状态（最大 `cash_idx/house_idx`）。
