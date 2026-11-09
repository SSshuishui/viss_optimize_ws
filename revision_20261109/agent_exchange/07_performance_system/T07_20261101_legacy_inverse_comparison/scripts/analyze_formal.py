#!/usr/bin/env python3
import csv,json,hashlib,datetime
from pathlib import Path
import numpy as np

def compare(a,b):
    a=np.asarray(a,dtype='f8');b=np.asarray(b,dtype='f8');d=a-b
    return {'relative_l2':float(np.linalg.norm(d)/np.linalg.norm(b)),'pearson':float(np.corrcoef(a,b)[0,1]),'rmse_over_reference_std':float(np.sqrt(np.mean(d*d))/np.std(b))}
def stats(x):
    a=np.array(x,dtype='f8')
    return {'raw':list(map(float,a)),'n':len(a),'median':float(np.median(a)),'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'iqr':float(np.quantile(a,.75)-np.quantile(a,.25)),'min':float(a.min()),'max':float(a.max())}
def main():
    root=Path(__file__).resolve().parent.parent;common=root.parent.parent/'01_common_operator'
    ref=np.load(common/'T01_20261101_production_replay/runs/day1_s10_g1_n8192/case.npz')
    fixed=common/'T01_20261101_legacy3d_lineage/runs/day1_direct_fixed'
    fixedN=np.fromfile(fixed/'raw_N_mode1.f32',dtype='<f4').astype('f8');fixedW=np.fromfile(fixed/'W_mode1.f64',dtype='<f8');fixedC=fixedN/fixedW
    natural=ref['direct_N']/ref['direct_W'];cell=ref['cell_direct_N']/ref['cell_W']
    rows=[];all_results=[];quality=[];stage=[]
    blocks=[['A','B','D','C'],['B','C','A','D'],['C','D','B','A'],['D','A','C','B']]
    for block,order in enumerate(blocks,1):
        for position,method in enumerate(order,1):
            folder=root/'runs'/f'formal_b{block}_{method}';d=json.loads((folder/'result.json').read_text());assert d['gate']['pass']
            for x in json.loads((folder/'manifest.json').read_text())['files']:
                p=folder/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],str(p)
            t=d['timing'];loop=t['inverse_loop_wall_s'];finalize=t['output_finalize_wall_s'];ids=np.fromfile(folder/'sample_ids.u32',dtype='<u4');assert np.array_equal(ids,ref['pixel_ids'])
            n=np.fromfile(folder/'sample_N.f32',dtype='<f4').astype('f8')
            w=np.fromfile(folder/'sample_W.u32',dtype='<u4') if method in ('C','D') else fixedW
            c=np.divide(n,w,out=np.zeros(8192,dtype='f8'),where=w>0)
            rows.append({'block':block,'position':position,'method':method,'inverse_loop_wall_s':loop,'output_finalize_wall_s':finalize,'loop_plus_finalize_wall_s':loop+finalize,'cache_load_decode_wall_s':t['cache_load_decode_wall_s'],'init_wall_s':t['init_wall_s'],'rawN_SHA256':d['gate']['full_rawN_sha256'],'sample_N_reference_L2':d['gate']['sample_N']['relative_l2']})
            q={'block':block,'method':method,'fixedDCF_diagnostic_N_over_W':compare(c,fixedC),'natural_diagnostic_N_over_W':compare(c,natural),'samecell_conditional_reference':compare(c,cell)};quality.append(q);all_results.append(d)
            sr=list(csv.DictReader((folder/'segments.csv').open()));assert len(sr)==10
            sumfield=lambda name:sum(float(x[name]) for x in sr)
            stage.append({'block':block,'method':method,'H2D_event_s':sumfield('h2d_event_s' if method in ('A','B') else 'input_h2d_cuda_s'),'weight_or_group_wall_s':sumfield('prep_wall_s' if method in ('A','B') else 'weight_or_group_wall_s'),'recon_wall_s':sumfield('recon_wall_s'),'recon_event_span_s':sumfield('recon_stream_span_s' if method in ('A','B') else 'recon_cuda_s'),'scope':'event spans are per-method diagnostics; B includes planner dependency intervals'})
    out=root/'metrics';out.mkdir(exist_ok=True)
    for name,data in [('formal_runs.csv',rows),('stage_sums.csv',stage)]:
        with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    compact=[{'method':x['method'],'round':x['block'],'inverse_loop_wall_s':x['inverse_loop_wall_s'],'normalize_finalize_wall_s':x['output_finalize_wall_s']} for x in rows]
    with (out/'formal_raw_values.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
    for category,pattern in [('gate','gate_*'),('warmup','warmup_*'),('resume_warmup','resume_warmup_*')]:
        excluded=[]
        for folder in sorted((root/'runs').glob(pattern)):
            d=json.loads((folder/'result.json').read_text());t=d['timing'];excluded.append({'category':category,'name':folder.name,'method':d['method'],'inverse_loop_wall_s':t['inverse_loop_wall_s'],'normalize_finalize_wall_s':t['output_finalize_wall_s']})
        if excluded:
            with (out/(category+'_raw_values.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(excluded[0]));w.writeheader();w.writerows(excluded)
    methods={}
    for method in 'ABCD':
        rr=[x for x in rows if x['method']==method]
        methods[method]={'inverse_loop_wall_s':stats([x['inverse_loop_wall_s'] for x in rr]),'loop_plus_finalize_wall_s':stats([x['loop_plus_finalize_wall_s'] for x in rr])}
    paired=[]
    for block in range(1,5):
        rr={x['method']:x for x in rows if x['block']==block}
        for slow,fast in [('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')]:
            paired.append({'block':block,'ratio':slow+'/'+fast,'inverse_loop_ratio':rr[slow]['inverse_loop_wall_s']/rr[fast]['inverse_loop_wall_s'],'loop_plus_finalize_ratio':rr[slow]['loop_plus_finalize_wall_s']/rr[fast]['loop_plus_finalize_wall_s']})
    with (out/'paired_ratios.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    result={'state':'completed','at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'methods':methods,'paired_ratios':paired,'quality_all_formal':quality,'all16_valid_formal_kept':True,'full_output_pixels':201326592,'precision_domain':'8192 fixed samples for quality, complete201M output finite/SHA per run','comparison_scope':'same cached V inverse-only; original3D raw DCF output and STS cell normalization are distinct scientific responses; no equal-accuracy/end-to-end/sky-recovery claim','timing_scope':'loop includes inverse preparation/H2D/method computation; cache/init/full scan excluded; STS normalize reported separately','no_locks_no_CI':True}
    (out/'final_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'methods':methods,'all16_kept':True},indent=2))
if __name__=='__main__':main()
