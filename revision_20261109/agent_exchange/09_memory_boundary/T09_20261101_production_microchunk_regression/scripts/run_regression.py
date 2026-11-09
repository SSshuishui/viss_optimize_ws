#!/usr/bin/env python3
"""Finite root-authorized actual-production queue. Default is CPU prepare only."""
import argparse,csv,datetime,fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import budget_helpers as budget
import resource_amendment as resource
ROOT=Path(__file__).resolve().parents[1];EXCHANGE=ROOT.parent.parent
UUID2='GPU-d26f44d0-c546-24e1-433c-49a053ecf0ee';UUID3='GPU-99fbd651-f080-aa22-391b-f33caa723578'
NAMES=('prefix','capacity','m0_default','m4_default','m4_small','CPU_finalize')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def art(p):return {'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);budget.save(p,v)
def sigterm(signum,frame):raise KeyboardInterrupt('Supervisor SIGTERM: stop owned process group and persist budget')
def protocol():return load(ROOT/'PROTOCOL.json')
def records(folder):return [{'path':str(p.relative_to(folder)),**art(p)} for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts]
def immutable_files():
 return [p for d in ('source','scripts','bin','references') for p in sorted((ROOT/d).rglob('*')) if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'PROTOCOL.json']
def prepare():
 p=protocol();binary=ROOT/'bin/production_microchunk_regression';inp=Path(p['input']['path'])
 if not binary.exists() or not inp.exists():
  v={'state':'local_source_prepared_pending_remote_build_input_identity','at':now(),'source_records':[{'path':str(v.relative_to(ROOT)),**art(v)} for v in immutable_files()],'binary_present':binary.exists(),'input_present':inp.exists()};save(ROOT/'preparation.json',v);return None
 inp_identity=art(inp);assert inp_identity=={k:p['input'][k] for k in ('bytes','sha256')},'original full raw B identity mismatch'
 frozen={'state':'frozen','scope':ROOT.name,'files':[{'path':str(v.relative_to(ROOT)),**art(v)} for v in immutable_files()],'input':{'path':str(inp),**inp_identity,'mtime_ns':inp.stat().st_mtime_ns,'inode':inp.stat().st_ino},'process_cases':list(NAMES),'program_budget_seconds':3600,'GPU_UUIDs':[UUID2,UUID3]}
 chain=resource.chain()
 if chain is not None:frozen['resource_chain']=chain;frozen['ancestor_case_bindings']=chain['ancestor_case_bindings']
 dest=ROOT/'execution_freeze.json'
 if dest.exists() and load(dest)!=frozen:
  assert chain is not None and sha(dest)==chain['parent_execution_freeze_sha256'],'frozen identity drift outside explicit one-time resource chain'
  save(dest,frozen)
 elif not dest.exists():save(dest,frozen)
 return sha(dest)
def verify_freeze():
 f=load(ROOT/'execution_freeze.json');assert f['scope']==ROOT.name and f['program_budget_seconds']==3600 and f['process_cases']==list(NAMES)
 for rec in f['files']:assert art(ROOT/rec['path'])=={k:rec[k] for k in ('bytes','sha256')},('frozen identity changed',rec['path'])
 inp=Path(f['input']['path']);assert inp.stat().st_size==f['input']['bytes'] and inp.stat().st_mtime_ns==f['input']['mtime_ns'] and inp.stat().st_ino==f['input']['inode'],'raw B identity/stat drift'
 if 'resource_chain' in f:assert resource.chain()==f['resource_chain']
 return sha(ROOT/'execution_freeze.json')
def authorize(marker,max_cases):
 a=load(marker);assert a['allow_execute'] is True and a['scope']==ROOT.name and a['execution_freeze_sha256']==verify_freeze()
 assert a['cumulative_process_wall_budget_seconds']==3600 and a['GPU_UUIDs']==[UUID2,UUID3]
 assert list(NAMES[:max_cases])==a['allowed_cases'][:max_cases] and set(NAMES[:max_cases]).issubset(a['allowed_cases'])
 mode=a['authorization_mode']
 if mode=='root_gate_only':assert max_cases<=2,'early root gate permission cannot run complete production'
 else:
  assert mode=='conditional_queue'
  q=Path(a['queue_root']);q=q if q.is_absolute() else EXCHANGE/q
  checker=q/'scripts/handoff_checks.py';support=q/'scripts/execution_support.py'
  assert sha(checker)==a['handoff_checker_sha256'] and sha(support)==a['execution_support_sha256'],'root-bound handoff code drift'
  sys.path.insert(0,str(checker.parent));spec=importlib.util.spec_from_file_location('production_queue_handoff',checker);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  m.verify_previous_completed_and_quiet(q,'production_30MHz',EXCHANGE)
 return a
def boundary(name,reserve=0):
 assert not (ROOT/'PAUSE').exists(),'PAUSE: no automatic restart'
 p=protocol();used=sum(v.stat().st_size for v in ROOT.rglob('*') if v.is_file());free=shutil.disk_usage(ROOT).free
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True,timeout=15)
 rows=[];fail=[]
 for index,uuid in ((2,UUID2),(3,UUID3)):
  raw=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip();row=next(csv.reader([raw]));contexts=[x for x in apps.splitlines() if x.split(',')[0].strip()==uuid];rows.append({'NVML':raw,'contexts':contexts});fail += [] if int(row[0])==index and row[1].strip()==uuid and not contexts and float(row[2])<=256 else [raw]
 save(ROOT/'boundaries'/f'{name}.json',{'at':now(),'GPU_rows':rows,'all_contexts':apps,'own_bytes':used,'next_case_reserved_bytes':reserve,'disk_free_bytes':free,'failures':fail})
 assert not fail,'GPU2/3 busy or identity mismatch; preserve state, do not wait or use GPU0/1'
 assert used+reserve<=p['temporary_storage_limit_bytes'] and free>=p['minimum_disk_free_bytes'],'storage guard including next-checkpoint reserve'
def manifest(folder):
 save(folder/'manifest.json',{'execution_freeze_sha256':sha(ROOT/'execution_freeze.json'),'files':records(folder)})
def verify_case(folder):
 m=load(folder/'manifest.json');current=verify_freeze()
 if m['execution_freeze_sha256']!=current:
  ancestors=load(ROOT/'execution_freeze.json').get('ancestor_case_bindings',{});binding=ancestors.get(folder.name);assert binding is not None and folder.name in ('prefix','capacity')
  assert m['execution_freeze_sha256']==binding['execution_freeze_sha256'] and art(folder/'manifest.json')==binding['manifest_artifact'] and art(folder/'result.json')==binding['result_artifact'],'only exact passed parent gate artifacts may use the explicit ancestor freeze'
 for v in m['files']:assert art(folder/v['path'])=={k:v[k] for k in ('bytes','sha256')}
 r=load(folder/'result.json');assert r['gate']['pass'] is True;return r
def gate(name,out):
 if name=='prefix':
  a,b,c=load(out/'clone_gate.json'),load(out/'prefix_micro_reference.json'),load(out/'prefix.json');assert a['pass'] and a['bitwise'] and a['all_half_baselines']==586432 and b['pass'] and b['bitwise'] and b['baselines']==8 and c['pass'] and c['reset_buffers_bitwise'];return {'pass':True,'scope':c['scope'],'pilot_cost':load(out/'pilot_cost.json')}
 m=load(out/'memory.json');assert m['pass'] and m['reserve_pass'] and len(m['cards'])==2 and all(c['minimum_free_bytes']>=2**31 for c in m['cards'])
 g=load(out/'group_gate.json');assert g['pass'] and g['half_baselines']==586432 and g['groups']==86668
 if name=='capacity':assert load(out/'capacity.json')['pass'];return {'pass':True,'actual_G2_full_state_allocation':True,'memory':m}
 z=load(out/'production.json');assert z['state']=='completed' and z['G']==2 and z['input_pixels']==z['output_pixels']==3221225472 and z['all_half_baselines']==586432 and not z['maps_written']
 for f in ('N','W','C'):
  d=load(out/f'full_{f}.json');assert d['pixels']==3221225472 and d['nonfinite']==0 and d['bytes']==12884901888
 assert (out/'sample_ids.u32').stat().st_size==(out/'sample_N.f32').stat().st_size==(out/'sample_W.u32').stat().st_size==(out/'sample_C.f32').stat().st_size==8192
 assert (out/'forward_raw.f32x2').stat().st_size==(out/'forward_consumed.f32x2').stat().st_size==4691456
 commit=load(out/'forward_committed.json');assert commit['execution_identity']==verify_freeze() and commit['mode']==z['mode'] and commit['G']==2 and commit['slab']==z['slab'] and commit['raw_sha256']==sha(out/'forward_raw.f32x2')
 cleanup=load(out/'checkpoint_cleanup.json');assert len(cleanup['registered_before_delete'])==2 and not list(out.glob('*.ck')) and not list(out.glob('*.tmp'))
 if name.startswith('m4_'):
  ref=load(out/'same_prefix_allrep_gate.json');assert ref['pass'] and ref['bitwise'] and ref['baselines']==86668 and ref['input_pixels']==1073741824
  assert sha(out/'same_prefix_allrep_gate.f64x2')==sha(ROOT/'references/MODE4_G3_p0_allrep_raw.f64x2')
 return {'pass':True,'actual_full_production_domain':True,'same_G':2,'all_half_baselines':586432,'all_output_finite':True,'all_group_identity_exact':True,'CPU_inverse_pending_until_budgeted_finalize':True}
def command(name,out):
 stage='prefix' if name=='prefix' else 'capacity' if name=='capacity' else 'full';mode=4 if name in ('prefix','m4_default','m4_small') else 0;slab=2**25 if name in ('prefix','m4_small') else 2**27
 return [str(ROOT/'bin/production_microchunk_regression'),f'--regression_stage={stage}',f'--regression_mode={mode}','--execution_identity='+verify_freeze(),'--references='+str(ROOT/'references'),'--pause_file='+str(ROOT/'PAUSE'),'--btag=30M','--nside=16384','--day=1','--day_start=1','--day_count=1','--segs=10','--segment_limit=1','--sample_count=2048','--blockage=1','--occ_mode=2','--write_output=0','--write_viss=0','--write_baseline_txt=0','--B_mode=bin','--C_mode=bin','--run_tag='+name,'--gen_gpu_index=0','--orbit_seed=42','--sky_dir='+str(Path(protocol()['input']['path']).parent),'--out_dir='+str(out),'--gpus='+UUID2+('' if stage=='prefix' else ','+UUID3),'--viss_tile_pix=256',f'--sky_slab_pix={slab}']
def launch(name,cmd,log):
 ledger=budget.load(ROOT);remaining=3600-ledger['used_process_wall_seconds'];cap=protocol()['program_process_limit_seconds'][name];assert remaining>=cap,'insufficient declared per-case reservation; no partial blind launch'
 env={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID'};env.pop('CUDA_VISIBLE_DEVICES',None);timeout=shutil.which('timeout');assert timeout
 return budget.run_owned(ROOT,name,cmd,log,env,cap,timeout)
def run_case(name):
 final=ROOT/'runs'/name
 if final.exists():return verify_case(final)
 assert not any(a['case']==name for a in budget.load(ROOT)['attempts']),'unsubmitted/failed prior attempt preserved; no automatic retry'
 assert not (ROOT/'attempts'/name).exists(),'unsubmitted case preserved; explicit audited resume only'
 boundary('before_'+name,protocol()['next_fullcase_storage_reserve_bytes']);out=ROOT/'attempts'/name;out.mkdir(parents=True)
 cmd=command(name,out);save(out/'command.json',{'command':cmd,'at':now(),'formal_timing':False});save(ROOT/'status.json',{'state':'running','case':name,'at':now()})
 rec=launch(name,cmd,out/'run.log');assert rec['returncode']==0,'nonzero process: preserve checkpoints/results, no retry'
 result={'case':name,'gate':gate(name,out),'execution':rec};save(out/'result.json',result);manifest(out);final.parent.mkdir(exist_ok=True);os.replace(out,final);boundary('after_'+name);return result
def pilot_forecast():
 c=load(ROOT/'runs/prefix/pilot_cost.json');estimate=1.5*48*(c['original_first33M_forward_s']+2*c['micro_first33M_forward_s'])+3*225+3*90+120;remaining=3600-budget.load(ROOT)['used_process_wall_seconds'];r={'at':now(),'core_conservative_estimate_seconds':estimate,'remaining_seconds':remaining,'formula':'1.5*48*(original33M+2*micro33M)+3*225inverse+3*90scan+120CPU','pass':estimate<=remaining,'resource_only_no_mode_or_tolerance_selection':True};save(ROOT/'metrics/PILOT_FORECAST.json',r);assert r['pass'],'conservative core cost exceeds remaining fixed budget; no auto extension'
def final_cpu():
 final=ROOT/'runs/CPU_finalize'
 if final.exists():return verify_case(final)
 assert not any(a['case']=='CPU_finalize' for a in budget.load(ROOT)['attempts']),'CPU finalize already attempted; no automatic retry'
 boundary('before_CPU_finalize');out=ROOT/'attempts/CPU_finalize';out.mkdir(parents=True);cmd=[sys.executable,str(ROOT/'scripts/finalize_regression.py')];rec=launch('CPU_finalize',cmd,out/'run.log');assert rec['returncode']==0
 qa=load(ROOT/'metrics/FINAL_QA.json');assert qa['state']=='passed' and qa['gate']['pass'];save(out/'result.json',{'case':'CPU_finalize','gate':{'pass':True},'execution':rec});manifest(out);os.replace(out,final)
 # CPU oracles are produced after the CUDA jobs; bind the added small evidence explicitly.
 for name in ('m0_default','m4_default','m4_small'):
  folder=ROOT/'runs'/name;r=load(folder/'result.json');r['gate']['CPU_inverse_pending_until_budgeted_finalize']=False;r['gate']['CPU_inverse_pass']=load(folder/'CPU_INVERSE_ORACLE.json')['state']=='passed';save(folder/'result.json',r);manifest(folder)
def main():
 signal.signal(signal.SIGTERM,sigterm)
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--conditional-marker',type=Path);ap.add_argument('--max-cases',type=int,default=6);a=ap.parse_args();assert 1<=a.max_cases<=6
 if not a.execute:print('CPU_PREPARE',prepare());return
 assert a.conditional_marker is not None
 with (ROOT/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_freeze();authorize(a.conditional_marker,a.max_cases);resource.apply_signed()
  if not (ROOT/'budget_state.json').exists():assert not any((ROOT/d).exists() and any((ROOT/d).iterdir()) for d in ('runs','attempts')),'missing ledger beside work'
  budget.load(ROOT);s=load(ROOT/'sessions.json') if (ROOT/'sessions.json').exists() else {'sessions':[]};s['sessions'].append({'at':now(),'controller_identity':budget.process_identity(os.getpid()),'command':sys.argv});save(ROOT/'sessions.json',s)
  for name in NAMES[:min(a.max_cases,5)]:
   run_case(name)
   if name=='capacity' and a.max_cases>2:pilot_forecast()
  if a.max_cases<6:save(ROOT/'status.json',{'state':'gate_phase_completed' if a.max_cases<=2 else 'partial_phase_completed','at':now(),'completed_cases':list(NAMES[:a.max_cases]),'remaining_work_preserved':True});return
  final_cpu();ledger=budget.load(ROOT);assert [v['case'] for v in ledger['attempts']]==list(NAMES) and all(v['state']=='finished' and v['returncode']==0 for v in ledger['attempts']);assert ledger['used_process_wall_seconds']<=3600
  for name in NAMES:verify_case(ROOT/'runs'/name)
  boundary('all_completed');retained=sum(v.stat().st_size for v in ROOT.rglob('*') if v.is_file());assert retained<=protocol()['final_storage_limit_bytes']
  save(ROOT/'status.json',{'state':'completed','at':now(),'all_cases':list(NAMES),'full_production_cases':3,'G':2,'input_output_pixels':3221225472,'all_half_baselines':586432,'final_QA':'metrics/FINAL_QA.json','process_wall_seconds':ledger['used_process_wall_seconds'],'own_retained_bytes':retained,'GPU_work_finished':True,'no_new450_no_multi_node':True})
  save(ROOT/'manifest.json',{'execution_freeze_sha256':sha(ROOT/'execution_freeze.json'),'files':records(ROOT)})
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'status.json',{'state':'paused_or_failed','at':now(),'error':repr(e),'outputs_budget_and_checkpoints_preserved':True,'automatic_retry':False});raise
