#!/usr/bin/env python3
"""No CUDA/NVML/SSH. Recovery and supervisor-exit guards with injected identities."""
import importlib.util,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
import budget_helpers as b
import run_regression as r
ROOT=Path(__file__).resolve().parents[1]
def rejected(fn):
 try:fn()
 except (RuntimeError,AssertionError):return True
 return False
checks={}
with tempfile.TemporaryDirectory() as t:
 p=Path(t);s=b.load(p,identity_fn=lambda pid:None,token_fn=lambda token:[]);checks['limit3600']=s['limit_seconds']==3600
 n,token=b.begin(p,'test',['true'],60,identity_fn=lambda pid:None,token_fn=lambda token:[],wall_clock=lambda:1000)
 checks['unknown_launch_blocks']=rejected(lambda:b.load(p,identity_fn=lambda pid:None,token_fn=lambda token:[],wall_clock=lambda:1001))
 s=json.loads((p/'budget_state.json').read_text());checks['unknown_PID_consumes_remaining']=s['used_process_wall_seconds']==3600 and s['manual_clearance_required'];before=(p/'budget_state.json').read_bytes();checks['restart_does_not_reset_unknown']=rejected(lambda:b.load(p,identity_fn=lambda pid:None,token_fn=lambda token:[])) and before==(p/'budget_state.json').read_bytes()
with tempfile.TemporaryDirectory() as t:
 p=Path(t);b.load(p);n,token=b.begin(p,'test',['true'],60,identity_fn=lambda pid:None,token_fn=lambda token:[],wall_clock=lambda:1000);b.set_pid(p,n,123,identity_fn=lambda pid:{'pid':123,'start_ticks':'1','boot_id':'test'})
 checks['active_PID_rejects_duplicate']=rejected(lambda:b.load(p,identity_fn=lambda pid:{'pid':123,'start_ticks':'1','boot_id':'test'},token_fn=lambda token:[],wall_clock=lambda:1001))
 s=b.load(p,identity_fn=lambda pid:None,token_fn=lambda token:[],wall_clock=lambda:1015);checks['dead_known_PID_conservative_charge']=s['used_process_wall_seconds']==15 and s['attempts'][0]['not_automatically_retried']
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'runs/x').mkdir(parents=True);checks['missing_ledger_beside_work_rejected']=rejected(lambda:b.load(p))
class FakeChild:
 pid=123;returncode=None
 def wait(self,timeout=None):raise KeyboardInterrupt('fixture')
 def poll(self):return self.returncode
with tempfile.TemporaryDirectory() as t:
 p=Path(t);child=FakeChild()
 try:b.run_owned(p,'interrupted',['true'],p/'log',{},60,'timeout',popen_factory=lambda *a,**kw:child,stop_fn=lambda c:setattr(c,'returncode',-15),identity_fn=lambda pid:{'pid':123,'start_ticks':'1','boot_id':'test'},token_fn=lambda token:[])
 except KeyboardInterrupt:pass
 s=json.loads((p/'budget_state.json').read_text());checks['KeyboardInterrupt_finally_charges_stops']=s['attempts'][0]['state']=='failed_or_interrupted' and s['attempts'][0]['returncode']==-15 and s['used_process_wall_seconds']>0
try:r.sigterm(signal.SIGTERM,None)
except KeyboardInterrupt:checks['SIGTERM_raises_for_finally']=True
with tempfile.TemporaryDirectory() as t:
 flags=['-isystem','/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1','-isysroot','/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk','-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib'] if sys.platform=='darwin' else []
 exe=Path(t)/'host_qa';z=subprocess.run(['clang++','-std=c++17','-O0',*flags,str(ROOT/'scripts/test_support.cpp'),'-lcrypto','-o',str(exe)],capture_output=True,text=True);assert z.returncode==0,z.stderr
 z=subprocess.run([str(exe),str(Path(t)/'fixture')],capture_output=True,text=True);assert z.returncode==0,z.stderr;checks['10_real_CPP_host_helper_fixtures']=True
checks['fixed_process_sequence']=list(r.NAMES)==r.protocol()['process_cases']
checks['caps_sum3600']=sum(r.protocol()['program_process_limit_seconds'].values())==3600
checks['SIGTERM_source_handler']='signal.signal(signal.SIGTERM,sigterm)' in (ROOT/'scripts/run_regression.py').read_text()
out={'state':'passed' if all(checks.values()) else 'failed','checks':checks,'host_fixture_stdout':z.stdout,'scope':'CPU-only injected recovery and actual host-helper C++ tests; not CUDA numeric/alloc tests'}
(ROOT/'metrics/CPU_ORCHESTRATION_QA.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert all(checks.values())
