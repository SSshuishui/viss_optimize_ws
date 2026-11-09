#!/usr/bin/env python3
"""Local-only validation and compact reports; never import CUDA or contact host."""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np

root=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
old=root/'manifest_CPU_preparation.json'
if not old.exists():
    shutil.copy2(root/'manifest.json',old)
    archive=root/'pre_GPU_evidence';archive.mkdir(exist_ok=True)
    for name in ('README.md','status.json','QA.json'):shutil.copy2(root/name,archive/name)
census=json.loads((root/'mask_statistics.json').read_text())
execution=json.loads((root/'GPU_execution.json').read_text())
assert execution['status']=='completed' and len(execution['cases'])==36
assert all(c['status']=='completed' for c in execution['cases'])
assert execution['GPU']['actual_uuid']=='GPU-d26f44d0-c546-24e1-433c-49a053ecf0ee'
assert execution['GPU']['positive_isign_gate']['relative_l2']<1e-10
driver_sha=sha(root/'source_versions/run_masked_roi_first_GPU_attempt.py')
cases=[];rows=[];raw=[];max_memory=0;max_direct_error=0;max_D=0
for meta in census['cases']:
    label=meta['case'];path=root/'gpu_results'/f'{label}.json'
    case=json.loads(path.read_text());assert case['status']=='completed' and case['attempt']==0
    assert case['identity']['driver_sha256']==driver_sha
    assert case['identity']['actual_uuid']==execution['GPU']['actual_uuid']
    outpath=path.with_suffix('.npz');assert sha(outpath)==case['output_sha256']
    with np.load(root/'cpu_reference'/f'{label}.npz',allow_pickle=False) as q:
        reference=q['reference_complex'];D=q['D']
    with np.load(outpath,allow_pickle=False) as q:
        assert np.array_equal(q['D'],D)
        assert np.isfinite(q['GPU_direct']).all()
        for eps in ('1e-06','1e-09'):assert np.isfinite(q['NUFFT_eps_'+eps]).all()
    with np.load(root/'cases'/f'{label}.npz',allow_pickle=False) as q:assert np.array_equal(q['D'],D)
    max_direct_error=max(max_direct_error,case['GPU_direct']['raw_error_vs_CPU']['relative_l2'])
    for method,stat in [('direct',case['GPU_direct']),('NUFFT_1e-06',case['results']['1e-06']),('NUFFT_1e-09',case['results']['1e-09'])]:
        assert stat['apply_wall']['n']==10 and stat['apply_events']['n']==10
        checkpoint=json.loads((root/'gpu_results'/f'{label}.attempt00.{method}.timing.json').read_text())
        assert checkpoint['warmups']==1 and checkpoint['completed_measured_calls']==10
        assert checkpoint['wall_s']==stat['apply_wall']['individual_s']
        assert checkpoint['events_s']==stat['apply_events']['individual_s']
        for index,(wall,event) in enumerate(zip(checkpoint['wall_s'],checkpoint['events_s']),1):
            assert np.isfinite(wall) and wall>0 and np.isfinite(event) and event>0
            raw.append(dict(case=label,method=method,repeat=index,wall_s=wall,event_s=event))
    direct=case['GPU_direct']['apply_wall']['median_s']
    direct_once=case['direct_input_H2D_s']+direct+case['direct_D2H_s']+case['GPU_direct']['CPU_post_D2H_normalization_s']
    for eps,result in case['results'].items():
        assert result['numerical_gate_pass'] and result['D_shared_exact']
        memory=max([m['occupied_bytes'] for m in result['observed_memory_marks']]+[result['memory_after_apply']['occupied_bytes']])
        assert memory<=16*1024**3;max_memory=max(max_memory,memory)
        apply=result['apply_wall']['median_s']
        once=result['accumulator_setup_s']+result['all_group_input_build_H2D_s']+result['all_group_plan_setpts_s']+apply+result['D2H_s']+result['CPU_post_D2H_normalization_s']
        rows.append(dict(case=label,face=meta['face'],pixels=meta['targets'],eps=eps,
            exact_mask_groups=meta['exact_mask_groups'],transform_groups=result['transform_group_count'],boundary_groups=meta['boundary_groups'],
            visible_pair_fraction=meta['visible_pair_fraction'],direct_apply_median_s=direct,NUFFT_apply_median_s=apply,
            direct_apply_over_NUFFT_apply=direct/apply,plan_setpts_s=result['all_group_plan_setpts_s'],
            NUFFT_group_input_build_H2D_s=result['all_group_input_build_H2D_s'],NUFFT_accumulator_setup_s=result['accumulator_setup_s'],
            NUFFT_D2H_s=result['D2H_s'],NUFFT_CPU_normalization_s=result['CPU_post_D2H_normalization_s'],
            direct_one_call_observed_excluding_common_setup_s=direct_once,NUFFT_one_call_observed_excluding_common_setup_s=once,
            direct_one_call_over_NUFFT_one_call=direct_once/once,
            CPU_mask_and_grouping_s=case['CPU_mask_build_grouping_s'],
            raw_complex_relativeL2=result['raw_error_vs_CPU']['relative_l2'],raw_complex_max_abs=result['raw_error_vs_CPU']['max_absolute'],
            dirty_relativeL2=result['dirty_error_vs_CPU']['relative_l2'],dirty_RMSE_over_ref_std=result['dirty_error_vs_CPU']['rmse_over_reference_std'],
            diagnostic_execute_s=result['separate_one_instrumented_diagnostic']['total_group_execute_event_s'],
            diagnostic_merge_s=result['separate_one_instrumented_diagnostic']['total_group_merge_event_s'],
            observed_max_occupied_GiB=memory/1024**3))
    cases.append(case)
assert len(rows)==72 and len(raw)==1080
for name,data in [('results_all36.csv',rows),('raw_apply_all1080.csv',raw),('plot_data.csv',rows)]:
    with (root/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(data[0]));writer.writeheader();writer.writerows(data)
summary=[]
for eps in ('1e-06','1e-09'):
    for domain,predicate in [('all36',lambda row:True),('boundary6',lambda row:row['boundary_groups']>0)]:
        selected=[r for r in rows if r['eps']==eps and predicate(r)]
        summary.append(dict(domain=domain,eps=eps,cases=len(selected),apply_faster_than_direct=sum(r['direct_apply_over_NUFFT_apply']>1 for r in selected),
            direct_over_NUFFT_apply_min=min(r['direct_apply_over_NUFFT_apply'] for r in selected),
            direct_over_NUFFT_apply_max=max(r['direct_apply_over_NUFFT_apply'] for r in selected),
            max_raw_complex_relativeL2=max(r['raw_complex_relativeL2'] for r in selected),
            max_dirty_relativeL2=max(r['dirty_relativeL2'] for r in selected),
            max_dirty_RMSE_over_reference_std=max(r['dirty_RMSE_over_ref_std'] for r in selected),
            plan_setpts_s_min=min(r['plan_setpts_s'] for r in selected),plan_setpts_s_max=max(r['plan_setpts_s'] for r in selected),
            one_call_faster_excluding_shared_setup=sum(r['direct_one_call_over_NUFFT_one_call']>1 for r in selected)))
inputs=np.load(root/'inputs_existing.npz',allow_pickle=False);a=inputs['rows']
angle=np.float32(np.float32(-2)*np.float32(np.pi))*a[:,2]
co=np.cos(angle);si=np.sin(angle)
rotated=np.asarray(a[:,11]*co-a[:,12]*si,dtype='f4').astype('f8')+1j*np.asarray(a[:,11]*si+a[:,12]*co,dtype='f4').astype('f8')
stored=a[:,13].astype('f8')+1j*a[:,14].astype('f8')
phase=dict(convention='cached consumed-n V, positive2pi*(u*l+v*m+w*n), no second phase correction',
    positive_isign_tiny_oracle=execution['GPU']['positive_isign_gate'],
    independent_group_algebra_16probe_relativeL2_max=2.372245762541965e-15,
    host_like_FP32_raw_nm1_to_consumed_n_rotation_relativeL2=float(np.linalg.norm(rotated-stored)/np.linalg.norm(stored)),
    host_rotation_interpretation='Post-run diagnostic using NumPy float32 trig and non-FMA arithmetic; small library/rounding differences mean this is not a bitwise production replay or a newly preregistered acceptance gate')
write(root/'phase_and_mask_validation.json',phase)
qa=dict(state='passed_all36',cases=36,formal_measured_apply_calls=1080,valid_observations_excluded=0,attempts_per_case=1,
    warmups_per_method=1,measured_calls_per_method=10,all_raw_timing_checkpoints_match=True,all_output_hashes_verified=True,
    all_D_exact=True,all_mask_group_reconstructions_exact=True,all_numerical_gates_passed=True,
    maximum_GPU_direct_vs_CPU_relativeL2=max_direct_error,observed_device_occupied_max_bytes=max_memory,
    observed_device_occupied_max_GiB=max_memory/1024**3,guard_GiB=16,
    memory_scope='Observed phase-boundary total-minus-free, not continuous absolute peak',
    first_runtime_error=False,driver_sha256=driver_sha)
write(root/'QA.json',qa)
write(root/'summary.json',dict(state='completed',scope='1120 actual day1 half-baselines,36 compact masked ROIs,notfullsphere',
    summaries=summary,phase=phase,QA=qa,shared_node=True,clocks_locked=False,
    one_call_cost_scope='Method input/H2D+plan/setpts+complete apply+D2H+CPU normalization; excludes shared CPU mask/group preparation, context and global process JIT',
    global_process_JIT_s=execution['GPU']['JIT_s']))
write(root/'status.json',dict(state='completed',CPU_cases=36,GPU_cases=36,formal_measured_calls=1080,new_forward=False,
    no_450day_or_multinode=True,all_results_retained=True,GPU_process_exited=True))
allrows=[r for r in rows if r['eps']=='1e-06']
report='''# Exact masked ROI external baseline: all36 completed

2026-11-01，GPU2；同卡同进程、共享节点、未锁频。实际1120条day1 half-baselines、现成fullsky consumed-n V；不重算forward。36个预声明ROI全部完成，无runtime错误、无数值失败、无重试/排除。每case的direct及两档cuFINUFFT各1热身+10正式，1080个正式apply原值均保留。

完整binary mask row精确分组；每组只计算其可见targets并GPU scatter相加。所有group execute、清零和scatter均计入完整masked apply。D与CPU direct逐位相同。它是有限ROI的exact mask factorization，不是生产fullsphere快速算法结论。

| 范围 | eps | apply快于direct的case | direct/NUFFT apply比范围 | 最大complex L2 | 分列单次成本较低case |
|---|---|---:|---:|---:|---:|
'''
for s in summary:report+=f"| {s['domain']} | {s['eps']} | {s['apply_faster_than_direct']}/{s['cases']} | {s['direct_over_NUFFT_apply_min']:.5f}–{s['direct_over_NUFFT_apply_max']:.5f} | {s['max_raw_complex_relativeL2']:.4g} | {s['one_call_faster_excluding_shared_setup']}/{s['cases']} |\n"
report+='''
eps1e-6仅8/36的apply快于透明FP64 direct，eps1e-9的apply全部更慢。按包含各自输入/H2D、plan、apply、D2H及归一化的分列单次成本口径，松/严精度分别8/36与2/36较低；这个口径排除共同CPU准备和context/JIT，不能冒充整个流程总时间。6个具有真正空间mask边界的case在两种口径下均更慢：仅apply时松精度约14–34倍、严格精度约26–57倍。全见/全遮行可合并为单变换的ROI更容易获益，但不能用它们代表含边界的case。该结果支持“mask兼容路线可执行且有精度控制”，不支持所有NUFFT不可行或生产STS比所有外部算法更快。

## Six boundary cases: every result retained

下表时间为ms，plan+setpts为单次观测，apply为n10 median。完整H2D、accumulator setup、D2H、CPU归一化与原值见CSV。

| Case | 非零groups | direct apply | eps1e-6 plan / apply | eps1e-9 plan / apply |
|---|---:|---:|---:|---:|
'''
for r in allrows:
    if not r['boundary_groups']:continue
    strict=next(x for x in rows if x['case']==r['case'] and x['eps']=='1e-09')
    report+=f"| {r['case']} | {r['transform_groups']} | {r['direct_apply_median_s']*1000:.3f} | {r['plan_setpts_s']*1000:.3f} / {r['NUFFT_apply_median_s']*1000:.3f} | {strict['plan_setpts_s']*1000:.3f} / {strict['NUFFT_apply_median_s']*1000:.3f} |\n"
report+=f'''
## Cost, phase, memory and acceptance

Setup与apply分列：`results_all36.csv`给出所有72个ROI/eps行，包括各group输入构造/H2D、plan+setpts、完整apply、D2H、CPU归一化，以及排除共同CPU mask/group准备和context/JIT后的单次调用成本。共享CPU准备和全进程JIT（{execution['GPU']['JIT_s']:.6f}s）另列；不能用apply比值冒充总成本。额外一次带事件的execute/merge诊断与正式十次分开，避免大量内部事件改变正式计时。

相位为n-frame正号，不额外旋转缓存V。独立tiny isign门禁L2={phase['positive_isign_tiny_oracle']['relative_l2']:.3g}；CPU1/16逐位相同，16方向独立group代数最坏L2=2.37e-15，GPU direct对CPU最坏L2={max_direct_error:.3g}。全部36case的两档NUFFT相对误差通过预声明1e-4/1e-7门限；这些门限仅覆盖所声明有限域。输入raw-nm1到consumed-n的host-like FP32旋转复核差{phase['host_like_FP32_raw_nm1_to_consumed_n_rotation_relativeL2']:.3g}，作为补充诊断记录，不冒充逐位或预声明门禁。

两档eps的最大dirty RMSE/reference std分别为{max(r['dirty_RMSE_over_ref_std'] for r in rows if r['eps']=='1e-06'):.5g}与{max(r['dirty_RMSE_over_ref_std'] for r in rows if r['eps']=='1e-09'):.5g}；未用range-normalized指标替代它们。

阶段边界观测设备占用最大{max_memory/1024**3:.6f}GiB，低于16GiB guard。没有连续采样暂态峰值。GPU2 UUID与preflight一致，case间设备身份相同；未宣称整节点独占或锁频。

原始36case JSON/NPZ及108个timing checkpoint保留；`raw_apply_all1080.csv`导出全部正式wall/event值，0删除。CPU准备manifest和初次GPU实际脚本均单独留档。CPU1/16耗时是单次正确性准备观测，未冒充CPU性能benchmark。新V来自实际fullsky cache，与旧无mask ROI的reduced-sky V不同，不混算新旧比值。

输出仅36个紧凑ROI；12face prefix和三档大小在统计前声明。face0坐标来自生产GPU prefix，其余采用冻结CPU production-style NEST方向；所有方法消费同一份有限精度坐标/point mask。这里没有外推fullsphere、全部1954680日基线、生产FP32最优direct或450天。
'''
(root/'REPORT.md').write_text(report)
storage=dict(state='completed_within_budget',scope='This batch directory,local mirrored files',retained_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),limit_bytes=100*1024**2,
    source_fullsky_copies=0,dense_mask_matrices_persisted=False,GPU_process_exited=True)
assert storage['retained_bytes']<storage['limit_bytes'];write(root/'storage_completion.json',storage)
files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts)
write(root/'manifest.json',dict(state='completed',files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],total_bytes=sum(p.stat().st_size for p in files)))
print(json.dumps(dict(cases=36,measured_calls=1080,summaries=summary,retained_bytes=storage['retained_bytes']),indent=2))
