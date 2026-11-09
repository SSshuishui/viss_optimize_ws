"""Persistent process-wall ledger; pure CPU helpers, no CUDA or NVML access."""
import json,os,signal,subprocess,time,uuid
from pathlib import Path
LIMIT=3600.0
OPEN={'launching','running','termination_pending'}
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%S%z',time.localtime())
def save(path,d):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def process_identity(pid):
 try:
  text=Path(f'/proc/{pid}/stat').read_text();rest=text[text.rfind(')')+2:].split()
  if rest[0] in ('Z','X'):return None
  return {'pid':int(pid),'start_ticks':rest[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
 except (FileNotFoundError,ProcessLookupError):return None
 except (PermissionError,IndexError) as e:raise RuntimeError('cannot verify owned child identity; refuse new launch') from e

def token_processes(token):
 """Inspect only own command lines for our public launch token, never environments."""
 hits=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=(p/'cmdline').read_bytes()
   if ('--owned-attempt-token='+token).encode() in args:
    identity=process_identity(int(p.name))
    if identity is not None:hits.append(identity)
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return hits

def _raw(root):return json.loads((Path(root)/'budget_state.json').read_text())
def load(root,identity_fn=process_identity,token_fn=token_processes,wall_clock=time.time):
 root=Path(root);path=root/'budget_state.json'
 if not path.exists():
  if any((root/f).exists() and any((root/f).iterdir()) for f in ('runs','geometry')):raise RuntimeError('budget ledger missing beside existing work; refuse automatic reset')
  save(path,{'limit_seconds':LIMIT,'used_process_wall_seconds':0.0,'attempts':[],'scope':'all GPU child process wall including failed launches, interrupts and supervisor termination; source analysis separate'})
 s=_raw(root)
 if s['limit_seconds']!=LIMIT:raise RuntimeError('persistent budget limit changed')
 charged=sum(float(v.get('process_wall_seconds_charged',0)) for v in s['attempts'])
 if abs(charged-s['used_process_wall_seconds'])>1e-6:raise RuntimeError('budget sum mismatch; refuse reset')
 if s.get('manual_clearance_required'):raise RuntimeError('unknown child launch identity: budget consumed and automatic resume blocked pending explicit audit')
 changed=False
 unknown_launch=False
 for v in s['attempts']:
  if v['state'] not in OPEN:continue
  expected=v.get('process_identity')
  if expected is not None and identity_fn(expected['pid'])==expected:raise RuntimeError('previous owned supervisor still active; no duplicate GPU launch')
  if token_fn(v['launch_token']):raise RuntimeError('owned child or supervisor launch token still active; no duplicate GPU launch')
  reserved=float(v['reserved_upper_bound_seconds']);elapsed=wall_clock()-float(v['started_epoch'])
  # A parent may die after Popen but before the PID identity is durably recorded.
  # Neither an empty token scan nor a small elapsed interval proves no child exists.
  if expected is None:
   charge=max(float(v.get('process_wall_seconds_charged',0)),float(v.get('remaining_at_start_seconds',LIMIT-s['used_process_wall_seconds'])))
   unknown_launch=True
  else:charge=max(float(v.get('process_wall_seconds_charged',0)),reserved if elapsed<0 else min(reserved,max(0.0,elapsed)))
  v.update(state='unknown_launch_requires_manual_clearance' if expected is None else 'orphaned_attempt_conservatively_charged',finished_at=stamp(),process_wall_seconds_charged=charge,wall_measurement='unknown launch consumes all budget available at start; identified dead child uses conservative elapsed upper bound',not_automatically_retried=True)
  changed=True
 if changed:
  s['used_process_wall_seconds']=sum(float(v.get('process_wall_seconds_charged',0)) for v in s['attempts']);s['remaining_seconds']=max(0.0,LIMIT-s['used_process_wall_seconds'])
  if unknown_launch:
   s['manual_clearance_required']=True;s['state']='blocked_unknown_launch_identity';s['manual_clearance_reason']='Popen may have started a child before PID/start_ticks/boot_id was committed; no automatic new launch or budget reset'
  save(path,s)
 if unknown_launch:raise RuntimeError('unknown child launch identity: entire remaining-at-start budget charged; automatic resume blocked pending explicit audit')
 return s

def begin(root,name,inner_command,wall_limit,identity_fn=process_identity,token_fn=token_processes,wall_clock=time.time):
 s=load(root,identity_fn,token_fn,wall_clock);remaining=LIMIT-s['used_process_wall_seconds']
 if wall_limit<=0 or wall_limit>remaining+1e-6:raise RuntimeError('attempt exceeds persistent budget remaining')
 token=uuid.uuid4().hex;v={'attempt_number':len(s['attempts'])+1,'case':name,'inner_command':inner_command,'state':'launching','started_at':stamp(),'started_epoch':wall_clock(),'reserved_upper_bound_seconds':float(wall_limit),'remaining_at_start_seconds':float(remaining),'launch_token':token}
 s['attempts'].append(v);save(Path(root)/'budget_state.json',s);return v['attempt_number'],token

def set_pid(root,number,pid,identity_fn=process_identity):
 s=_raw(root);v=s['attempts'][number-1];assert v['state'] in OPEN
 v.update(state='running',pid=int(pid),process_identity=identity_fn(pid));save(Path(root)/'budget_state.json',s)

def finish(root,number,wall,rc,error,terminated=True):
 s=_raw(root);v=s['attempts'][number-1]
 if v['state'] not in OPEN:raise RuntimeError('attempt already closed; refuse double charge')
 v.update(state=('finished' if error is None and rc==0 else 'failed_or_interrupted') if terminated else 'termination_pending',finished_at=stamp(),process_wall_seconds_charged=float(wall),wall_measurement='monotonic launch/wait/termination wall in finally',returncode=rc,error=error)
 s['used_process_wall_seconds']=sum(float(v.get('process_wall_seconds_charged',0)) for v in s['attempts']);s['remaining_seconds']=max(0.0,LIMIT-s['used_process_wall_seconds']);s['overage_seconds']=max(0.0,s['used_process_wall_seconds']-LIMIT);save(Path(root)/'budget_state.json',s)
 return s

def stop(child):
 if child is None or child.poll() is not None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:child.wait(timeout=2)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait(timeout=2)

def run_owned(root,name,inner_command,log_path,env,wall_limit,timeout_executable,popen_factory=subprocess.Popen,stop_fn=stop,identity_fn=process_identity,token_fn=token_processes):
 # GNU timeout survives a killed Python parent and keeps TERM+KILL within reservation.
 number,token=begin(root,name,inner_command,wall_limit,identity_fn,token_fn);command=[str(timeout_executable),'--signal=TERM','--kill-after=2s',f'{max(.1,wall_limit-2):.6f}s',*inner_command,'--owned-attempt-token='+token]
 child=None;rc=None;error=None;started=time.monotonic()
 try:
  with Path(log_path).open('w') as out:
   child=popen_factory(command,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True);set_pid(root,number,child.pid,identity_fn);rc=child.wait(timeout=max(.1,wall_limit+1))
 except BaseException as e:
  error=repr(e)
  try:stop_fn(child)
  finally:
   if child is not None:rc=child.returncode
  raise
 finally:
  wall=time.monotonic()-started;finish(root,number,wall,rc,error,terminated=(child is None or child.poll() is not None))
 return {'returncode':rc,'process_wall_s':wall,'supervised_command':command,'budget_attempt':number}
