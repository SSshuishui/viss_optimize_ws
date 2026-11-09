# shared_operator_plan_recon_fastpath

基于 `shared_operator_plan_stage2_balance` 的 Recon 热路径小优化版本。

这个目录保持当前 shared/operator-plan 的总体调度框架不变，不切换到 tile-owned recon，也不改变 Stage-2 block-cyclic 负载均衡策略。它只针对 nside=4096 及更高分辨率下占比最高的 `recon_3d_direct_tasklist_vv_real` 热路径做保守优化，适合作为当前主线版本继续测试。

## 优化内容

1. 新增 `apply_pair_weight_to_viss_half_kernel`，在进入 Recon 前把 `pair_weight_half` 预先乘入 `Viss_half`。
2. 从 Recon 内层循环中移除 `pair_weight_half` 读取、`pairw <= 0` 分支判断和逐像素乘权操作。
3. 将 `isfinite(Viss)` 检查从 `pixel × baseline` 内层循环移动到预加权 kernel 中，只对每条 half-baseline 检查一次。
4. 默认参数保持不变，便于和 `shared_operator_plan_stage2_balance` 做公平对比：
   - `OP_RECON_TASK_BL_VALUE=128`
   - `OP_RECON_PLAN_BL_CHUNK_VALUE=512`
   - `OP_RECON_TASK_TILE_PIX_VALUE=256`
5. 增加编译期参数，后续可以不改源码测试不同 Recon task 粒度。

## 已观测性能变化

以下结果来自 nsys kernel summary，用于说明优化方向，不等同于完整端到端 wall time。

| 场景 | 原 `shared_operator_plan_stage2_balance` | 本版本 | 变化 |
|---|---:|---:|---:|
| nside=512，1MHz，总 kernel time | 6.990 s | 6.048 s | 约 1.16× |
| nside=512，1MHz，Recon vv | 1.637 s | 0.891 s | 下降约 45.5% |
| nside=4096，10MHz，3 天，总 kernel time | 1306.619 s | 984.615 s | 约 1.33× |
| nside=4096，10MHz，3 天，Recon vv | 778.962 s | 458.478 s | 下降约 41.1% |

该结果说明，预加权和 Recon vv 热路径瘦身对大分辨率场景有效。当前瓶颈由原来的 Recon vv 单边主导，逐步转为 Stage-1 Viss 与 Stage-2 Recon vv 接近共同主导。

## 编译方式

默认编译方式与原版本一致：

```bash
nvcc ... -O3 -std=c++17 shared_operator_plan_stage2_balance.cu -o shared_operator_plan_stage2_balance
```

建议后续按以下顺序测试参数组合：

```bash
# 1. 只增大 Recon task baseline 宽度
nvcc ... -O3 -std=c++17 \
  -DOP_RECON_TASK_BL_VALUE=256 \
  shared_operator_plan_stage2_balance.cu -o shared_operator_plan_stage2_balance_bl256

# 2. 增大 Recon task baseline 宽度，同时增大 plan chunk
nvcc ... -O3 -std=c++17 \
  -DOP_RECON_TASK_BL_VALUE=256 \
  -DOP_RECON_PLAN_BL_CHUNK_VALUE=1024 \
  shared_operator_plan_stage2_balance.cu -o shared_operator_plan_stage2_balance_bl256_ch1024

nvcc ... -O3 -std=c++17 \
  -DOP_RECON_TASK_BL_VALUE=256 \
  -DOP_RECON_PLAN_BL_CHUNK_VALUE=2048 \
  shared_operator_plan_stage2_balance.cu -o shared_operator_plan_stage2_balance_bl256_ch2048
```

如果怀疑仍有非有限值进入 Recon，可打开调试检查：

```bash
-DOP_RECON_DEBUG_CHECK_FINITE=1
```

如果 DCF 中存在大量零权重 baseline，可测试零值跳过：

```bash
-DOP_RECON_SKIP_ZERO_WEIGHTED_VISS=1
```

## 结果一致性说明

本版本的结果应与 `shared_operator_plan_stage2_balance` 数值上非常接近，但不保证 bitwise 完全一致。原因是计算顺序从：

```cpp
pairw * (z.x * c - z.y * s)
```

调整为：

```cpp
(pairw * z.x) * c - (pairw * z.y) * s
```

建议使用 max/min、RMSE、主峰位置和图像主体结构进行对比，不建议直接比较二进制文件是否完全一致。
