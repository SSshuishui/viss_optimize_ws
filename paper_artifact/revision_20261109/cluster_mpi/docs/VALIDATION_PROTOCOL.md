# 集群版准备阶段验证协议

本协议区分“同一服务器的MPI进程集成验证”和“真实多节点实验”。当前准备阶段不租卡，不宣称A4000性能、网络带宽或跨节点scaling已验证。原始生产headers及main保持不变，MPI驱动位于独立目录。

## 运行前提

编译`build/cluster_sts_mpi`及不可变生产参考`build/production_reference`。Python只需要NumPy，不依赖healpy/scipy。由外层入口在启动前一次性确认目标GPU显存与其他任务情况；验证脚本不轮询GPU、不改时钟、不终止外部进程。

`tests/run_smoke_matrix.py`只在明确选定的一张物理GPU上运行1/2/3个MPI rank，并显式传`--allow-shared-gpu=1`。这种同卡共享用于正确性检查，时间仅是进度记录。MPI运行时和CUDA上下文的总显存应由实际启动环境预算，不将数组大小误称为进程总显存。

示例（参数按实际环境替换，不要把示例当已执行）：

```bash
python3 tests/run_smoke_matrix.py \
  --binary build/cluster_sts_mpi \
  --original-binary build/production_reference \
  --mpi-launcher deps/mpich/bin/mpiexec \
  --gpu 1 --output-dir validation/new_attempt_001
```

OpenMPI若需要超额CPU slots，可额外传`--mpi-extra=--oversubscribe`；该选项不能直接套用到MPICH。

验证器本身的CPU负控制可先运行`python3 tests/test_compare_runs.py`。初版七项覆盖有效不等长分片、局部shard损坏、错误平均归一化图、ownership重叠、输入身份改变、仅completed标签而缺文件及serial门禁失败。v2增加禁止无意混合策略、显式跨策略比较标记两项。2026-10-29本机NumPy环境已通过9/9，服务器MPI矩阵的实际结果仍须另外记录。

## 正向矩阵

- 固定确定性、非均匀正值binary32天空，另含三处强度不同的亮像素。输入按全局生产NEST pixel ID索引；几何方向全部由原CUDA kernel生成。
- Nside={8,16}，rep/exact两种重建，MPI ranks={1,2,3}，共12个MPI case。
- 每case为day1、time_count=8、segs=2、orbit_seed=42、grid_cell_scale=1、256像素native tile。Nside8有3个tile，2-rank分片不等长（512+256），专门检查非等长MPI gather与排序。
- 原始生产main用同一binary输入、相同参数和单GPU各运行一次，共4个original-reference case。rep对应occ_mode=2，exact对应occ_mode=0；occ_mode=1是first representative，不能当exact。
- MPI候选必须要求same-partition serial replay；这保留相同分片的FP32求和顺序，visibility/numerator/weight/map四项逐位相等。

## 完成不能只看退出码

`tests/compare_runs.py`逐项检查：

1. metrics完成状态，Nside/npix一致，rank数及native tile ownership范围正确，连续覆盖完整声明天空，无重复/遗漏/空rank。
2. 每日每rank的numerator、uint32 weight、normalized map文件大小和dtype正确，浮点全finite；按pixel ownership拼接后与全局三个数组逐位一致。
3. 对weight>0，map必须等于numerator/weight（最多4个FP32 ulp，容许CUDA fast division）；weight=0时两者为0；有非零有效信号，避免全零假通过。
4. MPI rank数改变后weight逐位相同；numerator/map的relative L2和peak-normalized最大绝对误差均不超过2e-4。该容差仅为短FP32积分的集成guard，不是论文精度预算；报告保留实际误差和逐位一致情况。
5. 单rank候选对原始生产单卡map：上述两种误差均不超过2e-5，并报告是否逐位一致。原始参考只有最终map，因此不声称原始生产版的numerator/weight也被独立导出验证。
6. 输入、执行binary、验证脚本及输出hash落盘；不同输入/参数不得直接混合成同一扩展对照。

`--dump-validation=1`可导出小问题的生产像素、signed几何、phase-corrected half visibility及分组元数据，便于后续独立CPU核查。当前本协议的独立集成参考是未改动生产main；不把共享生产kernel的serial replay称为独立高精度oracle。

## v2：每节点两卡及分层通信的准备验证

`python3 tests/test_launcher_plans.py`是纯CPU启动规划检查。通过可执行文件tripwire保证`--prepare-only`不调用MPI（包括版本探测）、nvidia-smi或CUDA程序。检查OpenMPI `--map-by ppr:2:node`、MPICH `-ppn 2`、1/2/3节点对应2/4/6 rank、GPU mask `0,1`；检查一GPU模式兼容、参数冲突、无效可见设备及两种策略矩阵。2026-10-29本机13/13项通过，无tripwire调用。实际两张GPU/节点的运行仍须独立验证。

GPU集成脚本可显式增加`--strategies=flat,hierarchical`，默认仍只执行flat初版协议：

- flat和hierarchical各跑Nside8/16×rep/exact×1/2/3 rank，共24个基础MPI case，并各自对照原始生产单卡输出，共8个原始参考case。
- hierarchical使用仅供小问题验证的`--validation-node-size=2`，在同一物理GPU上模拟逻辑分组。1/2/3 rank对应逻辑组大小[1]/[2]/[2,1]，特意覆盖不完整末组。
- Nside16的rep/exact再各加flat6与hierarchical6，共4个MPI case；hierarchical6覆盖三组×两rank。总计28个MPI case、8个原始参考。
- 每个相同策略的serial replay仍要求visibility/numerator/weight/map逐位一致。flat和hierarchical可能因两级FP32求和顺序不同而变化，显式跨策略比较保持2e-4相对L2及peak-normalized最大误差guard；不要求跨策略逐位一致。
- 两策略的相同rank数直接对照保存14份独立comparison JSON。普通比较若发现策略不一致则拒绝，必须显式授权跨策略比较，防止误将不同算法配置混为rank扩展。
- 所有真实host_count、GPU UUID及逻辑模拟标记必须留痕。逻辑三组×两rank不等于三台真实服务器或六张GPU，也不产生多节点性能结论。

## 负向配置

非2幂Nside、未知重建模式、缺失输入、负time_count、rank数超过native tile数、未经显式授权的同卡多rank都必须及时非零退出。collective hang/timeout属于失败，不能把超时视为配置被正常拒绝。timeout只终止本脚本启动的进程组。

任何门禁失败都保存日志及状态，用新attempt修复后重跑。不得覆盖旧失败、不按结果静默放宽阈值。若需要改协议，先在报告中解释误差来源、版本化旧阈值与新阈值，保留两份结果。

## 租用节点后才能确认的项目

1. A4000目标构建与真实驱动兼容、16GB每节点的内存可行域。
2. 实例间网络/端口、MPI启动、输入分发hash与网络通信正确性。无共享文件系统时，每节点的rank shard和profile只在该节点；须按metrics的rank_info从各节点收取，不能只同步rank0目录就声称全量原始结果已归档。
3. 相同问题、相同A4000、1/2/3节点的数值一致性、端到端耗时、归约/广播/构组/I/O分项。
4. 稳定资源下正式重复计时及加速效率。不同轨道日各自独立运行是任务吞吐，不等同同一成像任务的多节点strong scaling。

准备阶段的通过结果仅支持代码已构建及同机MPI集成可用；上述真实集群项目保持“待测”。
