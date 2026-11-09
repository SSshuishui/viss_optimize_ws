#!/usr/bin/env python3
"""Read compact CSV from exactly one production launch per stage."""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

root=Path(sys.argv[1])
stages={}
for stage in ['forward','representative_imaging']:
    path=root/'logs'/f'ncu_{stage}.csv'
    if not path.exists():
        stages[stage]={'state':'missing'}
        continue
    text=path.read_text()
    start=text.find('"ID","Process ID"')
    if start<0:
        stages[stage]={'state':'no_completed_metrics','log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        continue
    rows=list(csv.DictReader(io.StringIO(text[start:])))
    assert rows and len({r['ID'] for r in rows})==1
    metrics={r['Metric Name']:{'value':float(r['Metric Value'].replace(',','')),'unit':r['Metric Unit']} for r in rows}
    duration=metrics['gpu__time_duration.sum']['value']*1e-9
    stages[stage]={
        'state':'completed_one_kernel','kernel':rows[0]['Kernel Name'],
        'block':rows[0]['Block Size'],'grid':rows[0]['Grid Size'],
        'cc':rows[0]['CC'],'duration_s':duration,'metrics':metrics,
        'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if stage=='forward':
        stages[stage]['nominal_candidate_baseline_pixel_pairs']=195468*201326592
        stages[stage]['nominal_candidate_pairs_per_second']=(195468*201326592)/duration
        stages[stage]['throughput_scope']='Includes candidate pairs rejected by tile/point mask; not executed phase evaluations, FLOPs, or a fraction of FP32 hardware peak.'
data={
    'run_id':'T04_20261101_smallgate','profile_scope':'Unchanged production binary,10MHz,Nside4096,day1,segs10,physicalGPU2,one launch per stage,write_output0; targets killed after requested kernel. No full-day profile.',
    'binary_sha256_record':(root/'metrics/ncu_production_binary_sha256.txt').read_text().strip(),
    'ncu_version':'2024.1.1.0','stages':stages,
    'limits':['One representative launch is not a timing-repeat distribution or multi-GPU scaling test.','SM throughput is an Nsight pipeline metric, not achieved FLOP/s divided by advertised peak.','No SFU-specific metric or arithmetic instruction count was collected, so detailed compute sub-bottlenecks remain unresolved.','Profiler replay duration is separated from production-stage wall times.','Historical first two attempts were stopped; only the final direct-path completed CSVs are summarized.'],
}
out=root/'metrics/ncu_summary.json'
out.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
