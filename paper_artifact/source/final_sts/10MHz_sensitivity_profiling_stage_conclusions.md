# 10 MHz 敏感性实验与 Profiling 阶段性结论

## 1. 报告范围

本报告只总结当前结果中已经能够可靠得出的结论，覆盖：

1. segment 数量敏感性；
2. grid-cell size 敏感性；
3. 10 MHz、4×RTX 4090 的 profiling；
4. 默认配置结果的稳定性；
5. 当前可以写入论文的结论和建议表述。

本报告**暂不解释 Global 3D baseline 与 STS 的结果差异，也不对现有 Global 3D baseline 的正确实现方式作判断**。Global 3D baseline 的方法定义和后续修改，以后续重新确认的算法逻辑为准。现有 `comparison_results` 中只使用 `Representative STS vs Exact-group STS` 的结果；涉及 Global 3D 的两组数值暂不进入论文结论。

---

## 2. 当前实验配置

从结果文件可确认，本轮实验使用的核心配置为：

| 项目 | 配置 |
|---|---|
| 频率 | 10 MHz |
| HEALPix 分辨率 | Nside=4096 |
| 天空像素数 | 201,326,592 |
| 观测日 | day 1 |
| 时间采样数 | 69,810 |
| GPU | 4×RTX 4090 |
| 默认 segment 数量 | 10 |
| 默认 grid-cell scale | 1.0 |
| Exact reference | Exact group-wise STS |
| Final method | Representative-group STS |

需要特别说明：

- 本轮敏感性实验中的误差指标，是 **Representative-group STS 相对于同参数 Exact group-wise STS 的一致性指标**。
- 这些指标直接反映 representative-group occultation approximation 的影响。
- 它们不能单独表示 STS 相对于 Global 3D inverse 的总体误差。

---

# 3. Segment 数量敏感性

## 3.1 原始结果

固定：

```text
grid_cell_scale = 1.0
```

测试：

```text
segments = 5, 10, 20, 40
```

结果如下。

| Segments | Rep. visibility (s) | Rep. reconstruction (s) | Rep. total (s) | Exact reconstruction (s) | Recon. speedup | Pipeline speedup | Mean fit RMSE | Mean groups | Peak memory (MiB) | Pearson | Affine NRMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 37.923 | 19.079 | 57.003 | 255.283 | 13.38× | 5.15× | 0.094138 | 75,156.2 | 2369.25 | 0.998735 | 0.001354 |
| 10 | 39.565 | 19.422 | 58.989 | 256.478 | 13.21× | 5.02× | 0.044569 | 37,616.7 | 2267.25 | 0.998758 | 0.001298 |
| 20 | 41.180 | 19.526 | 60.709 | 257.903 | 13.21× | 4.93× | 0.043557 | 18,830.6 | 2207.25 | 0.998788 | 0.001290 |
| 40 | 53.569 | 19.684 | 73.317 | 258.385 | 13.13× | 4.25× | 0.043300 | 9,426.3 | 2177.25 | 0.998812 | 0.001291 |

其中：

- `Mean fit RMSE` 为 segment-local plane fitting 的平均残差，单位为 wavelength；
- Pearson 和 Affine NRMSE 均为 representative-group STS 相对于对应 exact-group STS 的全图结果；
- reconstruction speedup 为 exact reconstruction time / representative reconstruction time；
- pipeline speedup 为 exact total time / representative total time。

## 3.2 主要趋势

### 从 5 个 segment 增加到 10 个 segment

- 最终方法总时间从 57.003 s 增加到 58.989 s，只增加 **3.48%**；
- mean plane-fit RMSE 从 0.094138 降低到 0.044569，降低 **52.66%**；
- Affine NRMSE 从 0.001354 降至 0.001298，降低约 **4.14%**；
- Pearson 从 0.998735 略升至 0.998758；
- 峰值显存从 2369.25 MiB 降至 2267.25 MiB。

这说明将 observation timeline 从 5 段增加到 10 段，可以以很小的时间开销显著降低局部平面拟合残差。

### 从 10 个 segment 增加到 20 个 segment

- 总时间增加 **2.92%**；
- mean fit RMSE 只再降低 **2.27%**；
- Affine NRMSE 只从 0.001298 改善到 0.001290；
- representative reconstruction 时间基本不变。

继续增加 segment 后，平面拟合残差和 representative-group 一致性已经接近饱和。

### 从 10 个 segment 增加到 40 个 segment

- 总时间增加 **24.29%**；
- mean fit RMSE 只再降低 **2.85%**；
- Affine NRMSE 与 20 段几乎相同；
- visibility 时间从 39.565 s 增加到 53.569 s。

40 段时，更多的 segment 调度和固定开销开始明显增加整体运行时间，但没有带来相应的误差收益。

## 3.3 可得结论

当前结果支持继续使用：

```text
segments = 10
```

作为默认值。

原因不是单纯因为它运行最快，而是：

1. 相比 5 段，它显著降低了 plane-fit residual；
2. 相比 20 和 40 段，它保持了几乎相同的 representative-group 一致性；
3. 它避免了过多 segment 带来的调度和 visibility 固定开销；
4. 它在时间、局部几何拟合残差和显存之间形成了合理折中。

可以将 `segments=10` 描述为当前测试范围内的 **practical knee point**。

## 3.4 当前结果不能支持的结论

由于误差比较是：

```text
Representative STS(segs=x) vs Exact STS(segs=x)
```

所以目前不能据此断言：

```text
segs=10 相对于 Global 3D inverse 已经足够准确
```

本轮 segment sensitivity 能可靠支持的是：

- segment 数量对 plane-fit residual 的影响；
- segment 数量对运行时间和调度开销的影响；
- representative-group approximation 在不同 segment 设置下的稳定性。

---

# 4. Grid-cell size 敏感性

## 4.1 原始结果

固定：

```text
segments = 10
```

测试：

```text
grid_cell_scale = 0.75, 1.0, 1.25, 1.5
```

其中 scale=1.0 使用 segment sweep 中 `segment_s10` 的结果。

| Grid scale | Mean groups | Mean group size | Rep. reconstruction (s) | Rep. total (s) | Exact reconstruction (s) | Recon. speedup | Pearson | Affine NRMSE | Relative L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 48,723.1 | 4.198 | 24.959 | 64.447 | 268.477 | 10.76× | 0.999540 | 0.000924 | 0.029694 |
| 1.00 | 37,616.7 | 5.456 | 19.422 | 58.989 | 256.478 | 13.21× | 0.998758 | 0.001298 | 0.048873 |
| 1.25 | 30,530.5 | 6.730 | 15.858 | 55.548 | 249.414 | 15.73× | 0.996704 | 0.001689 | 0.080147 |
| 1.50 | 25,744.0 | 7.991 | 13.260 | 52.879 | 243.932 | 18.40× | 0.993859 | 0.002140 | 0.110479 |

## 4.2 主要趋势

结果呈现清晰且单调的性能—误差关系。

随着 grid-cell scale 增大：

1. occupied groups 数量持续下降；
2. 每个 group 中的平均 baseline 数量持续增加；
3. representative reconstruction 时间持续下降；
4. exact-group reconstruction 时间也有所下降；
5. representative-group 与 exact-group 的结果差异持续增大。

这说明 grid-cell size 的控制逻辑与预期一致：粗网格提高 baseline aggregation 程度，从而提高运行速度，但增大 representative-group approximation 的误差。

## 4.3 默认 scale=1.0 的位置

### 相比 scale=0.75

scale=1.0：

- representative reconstruction 从 24.959 s 降至 19.422 s，减少约 **22.2%**；
- total time 从 64.447 s 降至 58.989 s，减少约 **8.5%**；
- Pearson 从 0.999540 降至 0.998758；
- Affine NRMSE 从 0.000924 增至 0.001298。

也就是说，scale=1.0 获得了明显性能收益，同时误差仍维持在较低水平。

### 相比 scale=1.25

scale=1.25 相对 scale=1.0：

- total time 只再减少约 **5.8%**；
- reconstruction time 再减少约 **18.3%**；
- Affine NRMSE 从 0.001298 增至 0.001689，增加约 **30%**；
- Relative L2 从 0.048873 增至 0.080147。

### 相比 scale=1.5

scale=1.5 相对 scale=1.0：

- total time减少约 **10.4%**；
- Affine NRMSE 增加约 **64.8%**；
- Pearson 降至 0.993859；
- Relative L2 增至 0.110479。

粗化到 1.5 后，额外的整体时间收益已经有限，但结果差异增长明显。

## 4.4 可得结论

当前结果支持继续使用：

```text
grid_cell_scale = 1.0
```

作为默认值。

在测试范围内，它处于较明确的性能—误差折中位置：

- 0.75 更准确，但 reconstruction 成本更高；
- 1.25 和 1.5 更快，但结果差异上升更明显；
- 1.0 保持 Pearson 约 0.9988、Affine NRMSE 约 0.0013，同时获得约 13.2× 的 representative reconstruction speedup。

这一组结果可以作为论文中 approximation-control 论点的直接证据。

---

# 5. 默认配置下 Exact-group 与 Representative-group 的结果

默认配置：

```text
segments = 10
grid_cell_scale = 1.0
```

得到：

| 指标 | 数值 |
|---|---:|
| Exact reconstruction time | 256.478 s |
| Representative reconstruction time | 19.422 s |
| Reconstruction speedup | 13.21× |
| Exact total time | 296.110 s |
| Representative total time | 58.989 s |
| Whole-pipeline speedup | 5.02× |
| Pearson | 0.998758 |
| Raw NRMSE | 0.001300 |
| Affine NRMSE | 0.001298 |
| Relative L2 error | 0.048873 |
| Affine gain | 0.997673 |

该结果与论文已有五个代表日的平均结果相符：

- 当前 day-1 reconstruction speedup 为 13.21×；
- 已有五日平均为约 12.45×；
- 当前 day-1 Pearson 为 0.998758；
- 已有五日平均约为 0.998829；
- 当前 day-1 Affine NRMSE 为 0.001298；
- 已有五日平均约为 0.001330。

因此，本次新加入的 sensitivity instrumentation 没有改变原有主结论，day-1 结果也落在既有五日实验的合理范围内。

---

# 6. 10 MHz Profiling 结果

## 6.1 阶段级时间

正式 profiling 默认配置：

```text
segments = 10
grid_cell_scale = 1.0
occ_mode = representative
write_output = 0
```

测得：

| 阶段 | 时间 |
|---|---:|
| Visibility stage | 39.314 s |
| Reconstruction stage | 19.253 s |
| Segment-loop compute total | 58.569 s |
| Peak GPU memory | 2295.25 MiB |

按 wall-clock 阶段时间计算：

- visibility 占约 **67.1%**；
- reconstruction 占约 **32.9%**。

## 6.2 GPU kernel 构成

Nsight Systems 的 aggregate GPU kernel time 显示：

| Kernel | GPU kernel time 占比 |
|---|---:|
| `viss_partial_all_halfsym_tilecone` | 66.1% |
| `recon_seg_blockage_half_rep` | 33.9% |
| 其他所有 kernel | 接近 0% |

两个主要 kernel 几乎覆盖全部 GPU kernel 时间。

这说明：

1. 当前 10 MHz 最终方法是明显的 GPU-kernel-bound workload；
2. visibility kernel 是第一性能瓶颈；
3. representative reconstruction kernel 是第二性能瓶颈；
4. group construction、normalization、orbit 辅助 kernel 等不是主要瓶颈。

## 6.3 程序内部 critical-path 数据

程序记录：

| 项目 | 时间 |
|---|---:|
| Visibility kernel GPU critical path | 39.243 s |
| Visibility wall time | 39.314 s |
| Reconstruction kernel GPU critical path | 19.068 s |
| Reconstruction wall time | 19.253 s |
| Host reduction + phase correction | 0.065 s |
| Group construction | 0.028 s |

可以看出：

- visibility kernel 时间约占 visibility wall time 的 99%以上；
- reconstruction kernel 时间同样约占 reconstruction wall time的 99%；
- CPU reduction、phase correction 和 group construction 均不是当前瓶颈。

## 6.4 Host-device data movement

Nsight Systems 汇总结果：

| 操作 | 总数据量 | Aggregate GPU time |
|---|---:|---:|
| H2D memcpy | 1334.247 MB | 0.1511 s |
| D2H memcpy | 222.321 MB | 0.0089 s |
| D2D memcpy | 155.257 MB | 0.0006 s |
| CUDA memset | 19348.524 MB | 0.0191 s |

显式 H2D、D2H 和 D2D memcpy 的 aggregate time 合计约 0.161 s，相对于约 58.57 s 的 compute loop 非常小。

因此可以得出：

> 当前 10 MHz 运行不是 PCIe data movement 或 host reduction 受限，而主要由 visibility 和 reconstruction 两个 GPU kernel 主导。

## 6.5 关于 overlap 的限定

现有 profiling 结果没有证明大量的 device memcpy–kernel overlap，而且显式 memcpy 本身占比已经非常小。

因此论文中不宜把主要性能收益归因于“显著的 communication-computation overlap”。更准确的描述是：

- 实现使用了异步 stream、producer-worker 组织和可复用 buffer；
- 但在 10 MHz 默认配置下，测得的运行时间几乎全部由核心 GPU kernel 构成；
- 显式 host-device transfer 并不是主要瓶颈。

如果论文需要给出严格 overlap ratio，后续应对 producer、broadcast、visibility 和 reconstruction 添加明确 NVTX 区间，再按时间区间交集计算，而不能根据阶段累计时间推算。

## 6.6 输出 I/O

正式 Nsight profiling 关闭了输出，因此主要 profiling 时间不包含最终 dirty-map 写盘。

另一组开启输出的 check run 测得：

| 项目 | 时间 |
|---|---:|
| Output D2H | 0.0320 s |
| Total output stage | 0.2013 s |

在该机器和该次运行中，最终输出阶段相对于约 58.66 s 的 compute time 占比较低。

该结果只能描述当前服务器和当前本地存储条件，不宜泛化为所有存储系统上的固定结论。

---

# 7. 运行稳定性

三个相同或等价默认配置的运行时间为：

| 运行 | Total compute time |
|---|---:|
| Check run | 58.662 s |
| Nsight profile run | 58.569 s |
| Segment sweep，s=10 | 58.989 s |

三次运行最大差异不到 0.5 s，约低于 1%。

这说明：

1. 默认结果的复现性较好；
2. Nsight Systems profiling 没有造成明显的整体运行扰动；
3. sensitivity sweep 与单独默认运行的配置基本一致；
4. 本轮 profiling 数值可以用于论文中的阶段分析。

---

# 8. 当前可以写进论文的结论

## 8.1 Segment sensitivity

可以写：

> Increasing the number of temporal segments from 5 to 10 reduced the mean local-plane fitting residual by 52.7%, while increasing the representative pipeline runtime by only 3.5%. Further increasing the segment count to 20 or 40 produced only marginal reductions in the fitting residual and representative-group consistency error, whereas 40 segments increased total runtime by 24.3% relative to the 10-segment configuration. We therefore use 10 segments as the default practical trade-off.

需要避免写：

> 10 segments 已证明与 Global 3D inverse 误差足够小。

当前数据尚未支持这一句。

## 8.2 Grid-cell sensitivity

可以写：

> Grid-cell coarsening produced a monotonic performance–accuracy trade-off. Increasing the grid-cell scale from 0.75 to 1.5 reduced the mean number of occupied groups from 48,723 to 25,744 and reduced representative reconstruction time from 24.96 s to 13.26 s. At the same time, Pearson consistency against the exact group-wise STS reference decreased from 0.999540 to 0.993859, and affine-aligned NRMSE increased from 0.000924 to 0.002140. The default scale of 1.0 lies near the knee of this trade-off.

## 8.3 Profiling

可以写：

> Profiling shows that the 10 MHz execution is overwhelmingly GPU-kernel bound. The visibility and representative reconstruction stages account for 67.1% and 32.9% of the measured compute-loop wall time, respectively. In the aggregate Nsight kernel profile, the visibility and reconstruction kernels account for 66.1% and 33.9% of GPU kernel time. Explicit H2D, D2H, and D2D transfers together require approximately 0.161 s, compared with 58.57 s for the complete compute loop, indicating that host-device transfer is not the dominant bottleneck in this configuration.

## 8.4 默认方法

可以写：

> Under the default 10-segment and unit-grid-scale configuration, representative-group processing reduces day-1 reconstruction time from 256.48 s to 19.42 s, corresponding to a 13.21× reconstruction-stage speedup and a 5.02× whole-pipeline speedup relative to exact group-wise STS. The resulting dirty map has a Pearson correlation of 0.998758 and an affine-aligned NRMSE of 0.001298 against the exact-group result.

---

# 9. 建议加入论文的表格

## 表 A：Approximation sensitivity

可以将 segment 和 grid-cell 结果分成两个子表。

### A1. Segment sensitivity

保留以下列：

```text
Segments
Representative total time
Representative reconstruction time
Mean fit RMSE
Mean occupied groups
Pearson vs exact STS
Affine NRMSE vs exact STS
Peak GPU memory
```

### A2. Grid-cell sensitivity

保留以下列：

```text
Grid-cell scale
Mean occupied groups
Mean group size
Representative reconstruction time
Representative total time
Reconstruction speedup
Pearson vs exact STS
Affine NRMSE vs exact STS
```

## 表 B：Profiling breakdown

建议保留：

```text
Visibility kernel
Representative reconstruction kernel
Host visibility reduction + phase correction
Group construction
H2D
D2H
D2D
Output D2H
Output write
Peak GPU memory
```

对于 kernel，推荐同时给出：

- stage wall time；
- aggregate GPU kernel share；
- 是否在 critical path 上。

不要把所有 aggregate multi-GPU kernel time直接与单次 wall time相加比较，因为 Nsight aggregate kernel time会同时累加四张 GPU 上的并行执行时间。

---

# 10. 建议加入论文的图

优先建议两张图：

## 图 1：Grid-cell performance–accuracy trade-off

横轴：

```text
grid_cell_scale
```

左纵轴：

```text
representative reconstruction time
```

右纵轴：

```text
affine NRMSE 或 Pearson
```

该图趋势最清楚，也最能直接支撑 approximation-control 论点。

## 图 2：Segment sensitivity

横轴：

```text
segments
```

可以绘制：

- total runtime；
- mean fit RMSE；
- affine NRMSE。

这张图应突出：

- 5→10 的残差收益明显；
- 10→20/40 收益趋于饱和；
- 40 段的运行时间明显上升。

---

# 11. 当前不应写入论文的内容

在 Global 3D baseline 的算法逻辑重新确认之前，以下内容都暂不写入论文：

1. Global 3D 与 Exact STS 的 Pearson、NRMSE 和 relative L2；
2. Global 3D 与 Representative STS 的 Pearson、NRMSE 和 relative L2；
3. 对当前 Global 3D 结果差异来源的算法性解释；
4. Global 3D 相对 STS 的准确性结论；
5. Global 3D speedup 表格。

现阶段只保留相关原始文件，等待 baseline 方法重新确定并重跑。

---

# 12. 阶段性总判断

当前新增实验已经补强了论文中的两个关键部分。

第一，approximation sensitivity 不再只是概念性描述。实验明确展示了：

- segment 数量控制局部平面拟合残差和调度开销；
- grid-cell size 控制 baseline group 数量、运行时间和 representative-group 误差；
- 默认 `segments=10`、`grid_cell_scale=1.0` 具有可解释的实验依据。

第二，profiling 明确定位了系统瓶颈：

- 10 MHz 运行主要由 visibility 和 reconstruction 两个 GPU kernel 主导；
- host reduction、group construction 和显式 host-device transfer 都不是主要瓶颈；
- 继续优化时，优先级应放在两个核心 kernel，而不是进一步压缩已经很小的传输开销。

因此，目前可以确认写入论文的核心结论是：

> The default STS configuration is supported by a measured performance–accuracy sensitivity study, and the 10 MHz implementation is primarily limited by its two dominant GPU kernels rather than by host-side processing or explicit PCIe transfers.

Global 3D controlled baseline 仍作为独立待完成项处理，不与本报告中的已确认结论混合。
