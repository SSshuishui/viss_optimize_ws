#!/usr/bin/env python3
import argparse,datetime,hashlib,json,os
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);a=ap.parse_args();root=a.root.resolve()
    audit=[];removed=[]
    for folder in sorted((root/'runs').iterdir()):
        if not folder.is_dir():continue
        for f in json.loads((folder/'case_manifest.json').read_text())['files']:
            p=folder/f['path'];assert p.is_file() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
            audit.append({'path':str(p.relative_to(root)),**{k:f[k] for k in ('bytes','sha256')}})
            if p.name=='pair_weights_legacy.f32':removed.append(audit[-1])
    assert len(list((root/'runs').glob('*/case_manifest.json')))==14
    before=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    for f in removed:
        p=root/f['path'];assert p.name=='pair_weights_legacy.f32' and 'runs' in p.parts;p.unlink()
    after=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());fs=os.statvfs(root)
    report={'state':'passed','at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'cases':14,'validated_output_files':len(audit),'validated_outputs':audit,'removed_after_hash_and_science_validation':removed,'temporary_weights_removed_bytes':sum(x['bytes'] for x in removed),'retained_before_bytes':before,'retained_after_bytes':after,'filesystem_free_bytes':fs.f_bavail*fs.f_frsize,'batch_under_100MiB':after<100*1024**2,'scope':'only this batch temporary pair_weights deleted; all raw N/W, CPU oracle and failed outputs retained'}
    (root/'cleanup_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('validated_outputs','removed_after_hash_and_science_validation')},indent=2))
if __name__=='__main__':main()
