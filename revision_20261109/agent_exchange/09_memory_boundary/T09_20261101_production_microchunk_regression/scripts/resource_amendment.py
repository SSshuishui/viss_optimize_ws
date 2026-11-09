#!/usr/bin/env python3
"""One explicit resource-only revision. Never resets attempts, charges or results."""
from pathlib import Path
import hashlib,json,shutil
import budget_helpers as budget
ROOT=Path(__file__).resolve().parents[1]
OLD_FREEZE='171e114d31cbdbb94e31d0d40152b907b6c7a19d070effae9ab033a53c9b3498'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def art(p):return {'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def history():return ROOT/'resource_history/cap3000'
def archive(p,q):
 q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==p.read_bytes(),'resource history immutable'
 else:shutil.copyfile(p,q)
def numeric_unchanged(old):
 rows=[r for r in old['files'] if r['path'].startswith(('source/','bin/','references/'))]
 assert rows
 for r in rows:assert art(ROOT/r['path'])=={k:r[k] for k in ('bytes','sha256')},('numeric source/bin/reference changed',r['path'])
 return rows
def stage():
 oldpath=ROOT/'execution_freeze.json';assert sha(oldpath)==OLD_FREEZE
 old=load(oldpath);b=load(ROOT/'budget_state.json');assert b['limit_seconds']==3000 and [v['case'] for v in b['attempts']]==['prefix','capacity'] and all(v['state']=='finished' and v['returncode']==0 for v in b['attempts'])
 assert b['used_process_wall_seconds']==sum(v['process_wall_seconds_charged'] for v in b['attempts'])
 rows=numeric_unchanged(old);archive(oldpath,history()/'execution_freeze.json');archive(ROOT/'budget_state.json',history()/'budget_state.json')
 bindings={}
 for name in ('prefix','capacity'):
  folder=ROOT/'runs'/name;assert load(folder/'result.json')['gate']['pass'] and load(folder/'manifest.json')['execution_freeze_sha256']==OLD_FREEZE
  archive(folder/'manifest.json',history()/f'{name}_manifest.json');bindings[name]={'execution_freeze_sha256':OLD_FREEZE,'manifest_artifact':art(folder/'manifest.json'),'result_artifact':art(folder/'result.json')}
 proposal={'allow_resource_revision':True,'scope':ROOT.name,'old_execution_freeze_sha256':OLD_FREEZE,'old_budget_state':art(history()/'budget_state.json'),'old_budget_limit_seconds':3000,'old_used_process_wall_seconds':b['used_process_wall_seconds'],'new_budget_limit_seconds':3600,'additional_total_seconds':600,'future_process_cases':['m0_default','m4_default','m4_small','CPU_finalize'],'future_process_caps_seconds':{'m0_default':1100,'m4_default':1100,'m4_small':1100,'CPU_finalize':120},'future_reserved_seconds':3420,'remaining_after_reservation_seconds':3600-b['used_process_wall_seconds']-3420,'ancestor_case_bindings':bindings,'numeric_unchanged_records':rows,'original_full_input_identity':old['input'],'science_gates_G_IO_baselines_resolution_and_bin_unchanged':True,'reason':'Only original first33M resource pilot seen, no full MODE4 results: frozen forecast3032.162685408 exceeds remaining2940.291345. One explicit final resource revision to complete the same three predeclared cases; no error-based mode selection or tolerance change','no_further_automatic_resource_increase':True}
 dest=ROOT/'RESOURCE_AMENDMENT_TO_SIGN.json'
 if dest.exists():assert load(dest)==proposal
 else:budget.save(dest,proposal)
 return proposal
def chain():
 proposal=ROOT/'RESOURCE_AMENDMENT_TO_SIGN.json'
 if not proposal.exists():return None
 d=load(proposal);assert d['old_execution_freeze_sha256']==OLD_FREEZE and d['new_budget_limit_seconds']==3600 and d['future_reserved_seconds']==3420
 old=load(history()/'execution_freeze.json');assert sha(history()/'execution_freeze.json')==OLD_FREEZE;numeric_unchanged(old)
 assert art(history()/'budget_state.json')==d['old_budget_state']
 for name,r in d['ancestor_case_bindings'].items():assert name in ('prefix','capacity') and art(ROOT/'runs'/name/'manifest.json')==r['manifest_artifact'] and art(ROOT/'runs'/name/'result.json')==r['result_artifact']
 return {'authorization_path':'RESOURCE_AMENDMENT.json','authorization_artifact':art(proposal),'parent_execution_freeze_path':'resource_history/cap3000/execution_freeze.json','parent_execution_freeze_sha256':OLD_FREEZE,'parent_budget_path':'resource_history/cap3000/budget_state.json','parent_budget_artifact':d['old_budget_state'],'ancestor_case_bindings':d['ancestor_case_bindings'],'numeric_records_equal':True}
def apply_signed():
 c=chain();assert c is not None
 signed=ROOT/c['authorization_path'];assert art(signed)==c['authorization_artifact'],'root must sign exact proposed resource amendment'
 d=load(signed);assert d['allow_resource_revision'] and d['scope']==ROOT.name
 b=load(ROOT/'budget_state.json')
 if b['limit_seconds']==3600:
  assert b['resource_amendment']==c['authorization_artifact'];return
 assert b['limit_seconds']==3000 and art(ROOT/'budget_state.json')==c['parent_budget_artifact'],'no attempt/charge may change before resource-only migration'
 before_attempts=json.dumps(b['attempts'],sort_keys=True);used=b['used_process_wall_seconds'];b.update(limit_seconds=3600,remaining_seconds=3600-used,resource_amendment=c['authorization_artifact'],parent_budget_artifact=c['parent_budget_artifact'],resource_revision_count=1)
 assert json.dumps(b['attempts'],sort_keys=True)==before_attempts and b['used_process_wall_seconds']==used
 budget.save(ROOT/'budget_state.json',b)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--stage',action='store_true');ap.add_argument('--apply-signed',action='store_true');a=ap.parse_args()
 if a.stage:print(json.dumps(stage(),indent=2))
 if a.apply_signed:apply_signed();print('RESOURCE_ONLY_MIGRATION_APPLIED_NO_RESET')
