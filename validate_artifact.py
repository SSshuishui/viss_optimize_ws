#!/usr/bin/env python3
"""Validate the public source package without importing numerical modules or using CUDA."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REVISION = ROOT / 'revision_20261109'
EXCHANGE = REVISION / 'agent_exchange'
FOLLOWUP = EXCHANGE / 'followup_20261105'
GENERATED = {'point_only.cuh': FOLLOWUP / 'deploy/forward_ablation/make_point_kernel.py'}
REQUIRED = (
    REVISION / 'build_artifact.py',
    FOLLOWUP / 'deploy/five_day_timing/replay_3d_full.cu',
    FOLLOWUP / 'deploy/five_day_timing/replay_sts_full.cu',
    FOLLOWUP / 'deploy/sweep_repeats/replay_sts_sweep.cu',
    FOLLOWUP / 'deploy/forward_ablation/forward_ablation.cu',
    FOLLOWUP / 'direct450/source/direct_daily.cu',
    FOLLOWUP / 'direct450/runner/run_lane.py',
    FOLLOWUP / 'direct450/merge/merge_direct450.py',
    EXCHANGE / '01_common_operator/T01_20261101_production_replay/source/main_cache.cu',
    EXCHANGE / '01_common_operator/T01_20261101_production_replay/scripts/replay_common.hpp',
    EXCHANGE / '01_common_operator/T01_20261101_full_corrected_dcf_reference/scripts/replay_corrected_dcf_full.cu',
    EXCHANGE / '07_performance_system/T07_20261101_legacy_inverse_comparison/source/legacy_combined/viss_recon_kernel.cuh',
    EXCHANGE / '09_memory_boundary/T09_20261101_production_microchunk_regression/source/main_production_regression.cu',
    REVISION / 'cluster_mpi/src/main_cluster_mpi.cu',
    EXCHANGE / '05_baselines/T05_20261101_matched/scripts/direct.cpp',
    EXCHANGE / '05_baselines/T22_20261101_masked_roi/scripts/run_masked_roi.py',
)
GENERATED_DIRECTORIES = {'build', 'bin', 'run', 'cache', 'daily_cache', 'lanes', '__pycache__', '.git'}
CODE_SUFFIXES = {'.cu', '.cuh', '.hpp', '.h', '.cpp'}
LOCAL_MODULES = {'budget_helpers', 'build', 'compare_runs', 'mask_statistics', 'pair_stats',
                 'resource_amendment', 'run_batch', 'run_regression', 'run_short_gate'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    errors = []
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and not GENERATED_DIRECTORIES.intersection(p.relative_to(ROOT).parts)]
    for path in REQUIRED:
        if not path.is_file():
            errors.append({'file': str(path.relative_to(ROOT)), 'kind': 'required_file_missing'})
    python_count = bash_count = include_count = manifest_count = local_import_count = 0
    for path in files:
        relative = str(path.relative_to(ROOT))
        if path.is_symlink():
            errors.append({'file': relative, 'kind': 'source_symlink'})
        if path.suffix == '.py':
            python_count += 1
            try:
                compile(path.read_bytes(), relative, 'exec')
                for node in ast.walk(ast.parse(path.read_bytes())):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [item.name.split('.')[0] for item in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module.split('.')[0]]
                    for name in set(names).intersection(LOCAL_MODULES):
                        local_import_count += 1
                        candidates = []
                        for parent in path.parents:
                            if parent == ROOT.parent:
                                break
                            candidates += [parent / (name + '.py'), parent / 'scripts' / (name + '.py'),
                                           parent / 'tests' / (name + '.py')]
                        if not any(candidate.is_file() for candidate in candidates):
                            errors.append({'file': relative, 'kind': 'local_python_import_missing', 'module': name})
            except (SyntaxError, UnicodeError) as error:
                errors.append({'file': relative, 'kind': 'python_compile', 'detail': str(error)})
        if path.suffix == '.sh':
            bash_count += 1
            result = subprocess.run(['bash', '-n', relative], cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                errors.append({'file': relative, 'kind': 'bash_syntax', 'detail': result.stderr.strip()})
    headers = [p for p in files if p.suffix in CODE_SUFFIXES]
    by_name = {}
    for path in headers:
        by_name.setdefault(path.name, []).append(path)
    for path in headers:
        for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(), re.M):
            include_count += 1
            local = path.parent / name
            if local.is_file():
                continue
            if '/' not in name and (name in by_name or (name in GENERATED and GENERATED[name].is_file())):
                continue
            errors.append({'file': str(path.relative_to(ROOT)), 'kind': 'quoted_include_missing', 'include': name})
    manifests = [REVISION / 'PUBLIC_SOURCE_FILES.json']
    manifests += sorted(REVISION.glob('**/PUBLIC_SOURCE_FILES.json'))
    for manifest in dict.fromkeys(manifests):
        if not manifest.is_file():
            continue
        value = json.loads(manifest.read_text())
        for row in value['files']:
            target = ROOT / row['target']
            expected = row.get('published_sha256') or row.get('target_sha256') or row['sha256']
            manifest_count += 1
            if not target.is_file() or digest(target) != expected:
                errors.append({'file': row['target'], 'kind': 'published_source_hash_mismatch'})
    report = {'schema': 'artifact_source_validation_v1', 'state': 'passed' if not errors else 'failed',
              'required_files': len(REQUIRED), 'python_sources_compiled': python_count,
              'shell_scripts_syntax_checked': bash_count, 'quoted_includes_checked': include_count,
              'local_python_imports_checked': local_import_count,
              'source_manifest_entries_checked': manifest_count, 'errors': errors,
              'scope': 'Static package, Python syntax compilation, bash -n, quoted local/header search dependencies and published source fingerprints.',
              'cuda_compilation': 'Use revision_20261109/build_artifact.py on Linux with CUDA. This validator starts no CUDA runtime or GPU experiment.'}
    (REVISION / 'VALIDATION_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
