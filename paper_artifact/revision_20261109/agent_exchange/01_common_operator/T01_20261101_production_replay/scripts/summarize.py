#!/usr/bin/env python3
"""CPU-only final checks: fixed reference, rotation, and sampling density."""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np

def normalized(path,prefix):
    n=np.fromfile(path/(prefix+'_N.f64'),dtype='<f8');w=np.fromfile(path/'natural_W.f64',dtype='<f8')
    return np.divide(n,w,out=np.zeros_like(n),where=w>0)
def all_metrics(data,pair):return data['comparisons'][pair]['regions']['all']
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    runs=args.root/'runs';entries=[];by_name={}
    for path in sorted(runs.glob('day*')):
        if not (path/'metrics.json').exists():continue
        data=json.loads((path/'metrics.json').read_text());by_name[path.name]=data
        row={'case':path.name,'samples':data['samples'],'oracle_pass':data['oracle']['pass']}
        for pair in ('plane_vs_direct','quant_member_vs_direct','cell_direct_vs_direct','rep_vs_direct','centroid_vs_direct','rep_vs_cell_direct','centroid_vs_cell_direct','rep_vs_groupmean','centroid_vs_groupmean'):
            if pair not in data['comparisons']:continue
            a=all_metrics(data,pair)
            for key in ('raw_relative_l2','pearson','global_affine_rmse_over_ref_std'):row[pair+'_'+key]=a[key]
        entries.append(row)
    # Natural-sampling reference stays fixed across all segment/grid settings.
    baseline=runs/'day1_s10_g1_n4096';base_ids=np.fromfile(baseline/'ids.u32',dtype='<u4');base=normalized(baseline,'direct')
    fixed=[]
    for path in sorted(runs.glob('day1_*n4096')):
        if 'rot' in path.name:continue
        ids=np.fromfile(path/'ids.u32',dtype='<u4');out=normalized(path,'direct')
        delta=float(np.linalg.norm(out-base)/max(np.linalg.norm(base),1e-300))
        fixed.append({'case':path.name,'ids_exact':bool(np.array_equal(ids,base_ids)),'natural_direct_relative_l2_vs_default':delta,'pass':bool(np.array_equal(ids,base_ids) and delta<1e-12)})
    rotated=runs/'day1_s10_g1_rot037_n4096';rotation_out=normalized(rotated,'direct')
    delta=float(np.linalg.norm(rotation_out-base)/max(np.linalg.norm(base),1e-300))
    rotation={'angle_about_y_rad':.37,'ids_exact':bool(np.array_equal(np.fromfile(rotated/'ids.u32',dtype='<u4'),base_ids)),
        'natural_direct_relative_l2':delta,'natural_weight_counts_pixels_changed':int(np.count_nonzero(np.fromfile(rotated/'natural_W.f64',dtype='<f8')!=np.fromfile(baseline/'natural_W.f64',dtype='<f8'))),
        'pass':bool(delta<1e-3),'criterion':'exact natural operator invariant to rotation within1e-3 relativeL2, allowing frozen FP32 transform/predicate rounding',
        'default_rep_vs_natural':all_metrics(by_name[baseline.name],'rep_vs_direct'),
        'rotated_rep_vs_natural':all_metrics(by_name[rotated.name],'rep_vs_direct')}
    density=[]
    for day in (1,450):
        small=by_name[f'day{day}_s10_g1_n8192'];dense=by_name[f'day{day}_s10_g1_n32768'];items={}
        for pair in ('rep_vs_direct','rep_vs_cell_direct'):
            a=all_metrics(small,pair);b=all_metrics(dense,pair)
            items[pair]={'sample8192':a,'sample32768':b,
                'delta_pearson':b['pearson']-a['pearson'],'delta_relative_l2':b['raw_relative_l2']-a['raw_relative_l2'],
                'scope':'two independently stratified direction sets, not a full-sky error guarantee; preserve observed difference'}
        density.append({'day':day,'comparisons':items})
    result={'cases':entries,'fixed_natural_reference_checks':fixed,'rotation_exact_control':rotation,'sampling_density':density,
        'all_case_oracles_pass':all(row['oracle_pass'] for row in entries),
        'scope':'full production input sky and all single-day baselines; direction-sampled image references. Cell-direct changes with grouping and is an attribution diagnostic; fixed natural direct is the common scientific reference.'}
    (args.root/'final_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if entries:
        columns=sorted(set().union(*(row.keys() for row in entries)))
        with (args.root/'matrix.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(entries)
    print(json.dumps({'cases':len(entries),'all_oracles_pass':result['all_case_oracles_pass'],
        'fixed_reference_pass':all(r['pass'] for r in fixed),'rotation':rotation},indent=2))
    if not result['all_case_oracles_pass'] or not all(r['pass'] for r in fixed) or not rotation['pass']:raise RuntimeError('final scientific control failed')
if __name__=='__main__':main()
