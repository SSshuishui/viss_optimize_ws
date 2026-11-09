#!/usr/bin/env python3
"""CPU-only exact preservation and rejection tests for the one resource revision."""
import json,shutil,tempfile
from pathlib import Path
import budget_helpers as b
import resource_amendment as r
real=r.ROOT;checks={};chain=r.chain();assert chain is not None
def reject(fn):
 try:fn()
 except (AssertionError,RuntimeError,FileNotFoundError):return True
 return False
with tempfile.TemporaryDirectory() as t:
 root=Path(t)/real.name;root.mkdir()
 for d in ('source','bin','references','resource_history','runs/prefix','runs/capacity'):shutil.copytree(real/d,root/d)
 for name in ('RESOURCE_AMENDMENT_TO_SIGN.json','budget_state.json'):shutil.copyfile(real/name,root/name)
 r.ROOT=root;before=(root/'budget_state.json').read_bytes();old=json.loads(before)
 checks['unsigned_refused']=reject(r.apply_signed) and before==(root/'budget_state.json').read_bytes()
 shutil.copyfile(root/'RESOURCE_AMENDMENT_TO_SIGN.json',root/'RESOURCE_AMENDMENT.json');r.apply_signed();new=json.loads((root/'budget_state.json').read_text())
 checks['attempts_and_used_exact']=new['attempts']==old['attempts'] and new['used_process_wall_seconds']==old['used_process_wall_seconds']
 checks['only_600_added']=new['limit_seconds']==3600 and old['limit_seconds']==3000 and new['remaining_seconds']==3600-old['used_process_wall_seconds']
 checks['parent_ledger_exact_archived']=r.art(root/'resource_history/cap3000/budget_state.json')==chain['parent_budget_artifact']
 once=(root/'budget_state.json').read_bytes();r.apply_signed();checks['migration_idempotent_no_reset']=once==(root/'budget_state.json').read_bytes()
 checks['new_ledger_math_accepts']=b.load(root)['used_process_wall_seconds']==old['used_process_wall_seconds']
 q=root/'source/checkpoint_io.hpp';q.write_bytes(q.read_bytes()+b'\n');checks['numeric_source_drift_rejected']=reject(r.chain)
 r.ROOT=real
checks['ancestor_only_two_gates']=set(chain['ancestor_case_bindings'])=={'prefix','capacity'}
checks['future_caps3420']=sum(json.loads((real/'RESOURCE_AMENDMENT_TO_SIGN.json').read_text())['future_process_caps_seconds'].values())==3420
out={'state':'passed' if all(checks.values()) else 'failed','checks':checks,'scope':'CPU resource-only chain; no CUDA/new numerical results; old case manifests unchanged'}
(real/'metrics/CPU_RESOURCE_CHAIN_QA.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert all(checks.values())
