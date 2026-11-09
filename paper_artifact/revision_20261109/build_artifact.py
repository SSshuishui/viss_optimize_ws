#!/usr/bin/env python3
"""Compile revision components without running GPU experiments or reading sky data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXCHANGE = ROOT / 'agent_exchange'
FOLLOWUP = EXCHANGE / 'followup_20261105'
PRODUCTION = EXCHANGE / '01_common_operator/T01_20261101_production_replay'
LEGACY = EXCHANGE / '07_performance_system/T07_20261101_legacy_inverse_comparison'
CORRECTED = EXCHANGE / '01_common_operator/T01_20261101_full_corrected_dcf_reference'
COMPONENTS = ('five_day', 'sweep', 'forward', 'direct450', 'mature_direct', 'corrected_dcf', 'cpu_nufft', 'microchunk', 'mpi')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen(manifest, folder):
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = folder / name.lstrip('*')
        if digest(path) != expected:
            raise RuntimeError(f'Frozen source SHA mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=COMPONENTS, action='append', help='Repeat to select components. Default builds all.')
    parser.add_argument('--arch', choices=('sm_75', 'sm_86', 'sm_89'), default=os.environ.get('CUDA_ARCH', 'sm_89'))
    parser.add_argument('--nvcc', default=os.environ.get('NVCC', 'nvcc'))
    parser.add_argument('--host-cxx', default=os.environ.get('CUDA_HOST_CXX', 'g++'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    selected = args.component or COMPONENTS
    commands = []
    outputs = []
    environment = dict(os.environ, NVCC=args.nvcc, CUDA_ARCH=args.arch, ARCH=args.arch,
                       CUDA_HOST_CXX=args.host_cxx, CXX=args.host_cxx)
    flags = [args.nvcc, '-ccbin', args.host_cxx, '-O3', '--use_fast_math', '-lineinfo',
             '-std=c++17', '-arch=' + args.arch, '-Xcompiler', '-fopenmp']

    def execute(command, output=None):
        commands.append([str(x) for x in command])
        print(shlex.join(commands[-1]), flush=True)
        if not args.dry_run:
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(commands[-1], cwd=ROOT, env=environment, check=True)
        if output is not None:
            outputs.append(output)

    def cuda(source, output, includes=(), defines=(), crypto=True):
        command = flags + ['-D' + define for define in defines]
        for folder in includes:
            command += ['-I', folder]
        command += [source, '-o', output]
        if crypto:
            command.append('-lcrypto')
        execute(command, output)

    for component in selected:
        print(f'Building {component}', flush=True)
        if component == 'five_day':
            task = FOLLOWUP / 'deploy/five_day_timing'
            identity = json.loads((task / 'input_identities.json').read_text())
            for item in identity['header_files']:
                if digest(EXCHANGE / item['path']) != item['sha256']:
                    raise RuntimeError('Frozen followup header changed: ' + item['path'])
            cuda(task / 'replay_3d_full.cu', task / 'build/replay_legacy_combined_full',
                 (LEGACY / 'source/legacy_combined',), ('LEGACY_FASTPATH=1',))
            cuda(task / 'replay_sts_full.cu', task / 'build/replay_sts_full',
                 (PRODUCTION / 'source', PRODUCTION / 'scripts'))
        elif component == 'sweep':
            task = FOLLOWUP / 'deploy/sweep_repeats'
            cuda(task / 'replay_sts_sweep.cu', task / 'build/replay_sts_sweep',
                 (PRODUCTION / 'source', PRODUCTION / 'scripts'))
        elif component == 'forward':
            task = FOLLOWUP / 'deploy/forward_ablation'
            check_frozen(task / 'FROZEN_SHA256SUMS', task)
            execute([sys.executable, task / 'make_point_kernel.py'])
            cuda(task / 'forward_ablation.cu', task / 'build/forward_ablation',
                 (task / 'frozen', task / 'build'), crypto=False)
            if not args.dry_run:
                for compiler, filename in ((args.nvcc, 'nvcc_version.txt'), (args.host_cxx, 'host_cxx_version.txt')):
                    version = subprocess.run([compiler, '--version'], capture_output=True, text=True, check=True)
                    (task / 'build' / filename).write_text(version.stdout)
                (task / 'build/build_command.txt').write_text(shlex.join(commands[-1]) + '\n')
                inputs = [task / name for name in ('forward_ablation.cu', 'make_point_kernel.py', 'build.sh',
                          'run.sh', 'summarize.py', 'FROZEN_SHA256SUMS')]
                inputs += sorted((task / 'frozen').glob('*'))
                inputs += [task / 'build' / name for name in ('point_only.cuh', 'kernel_change.diff',
                           'nvcc_version.txt', 'host_cxx_version.txt', 'build_command.txt', 'forward_ablation')]
                (task / 'build/BUILD_SHA256SUMS').write_text(''.join(
                    digest(path) + '  ' + str(path.relative_to(task)) + '\n' for path in inputs))
        elif component == 'direct450':
            task = FOLLOWUP / 'direct450'
            check_frozen(task / 'source/FROZEN_SHA256SUMS', task / 'source')
            generated = task / 'build/forward_uuid.cu'
            if not args.dry_run:
                generated.parent.mkdir(exist_ok=True)
            if not generated.exists():
                execute([sys.executable, task / 'prepare_forward.py', PRODUCTION / 'source/main_cache.cu', generated])
            elif not args.dry_run:
                # Re-derive in memory and verify the existing generated source.
                import tempfile
                with tempfile.TemporaryDirectory(dir=generated.parent) as scratch:
                    expected = Path(scratch) / 'expected.cu'
                    subprocess.run([sys.executable, str(task / 'prepare_forward.py'),
                                    str(PRODUCTION / 'source/main_cache.cu'), str(expected)], check=True)
                    if digest(generated) != digest(expected):
                        raise RuntimeError('Generated UUID forward source changed')
            cuda(generated, task / 'bin/forward_uuid', (PRODUCTION / 'source', PRODUCTION / 'scripts'), crypto=False)
            cuda(task / 'source/direct_daily.cu', task / 'bin/direct_daily')
        elif component == 'mature_direct':
            cuda(LEGACY / 'scripts/replay_3d_full.cu', LEGACY / 'bin/replay_legacy_combined_full', defines=('LEGACY_FASTPATH=1',))
            cuda(LEGACY / 'scripts/replay_3d_full.cu', LEGACY / 'bin/replay_controlled_fixed_full', defines=('LEGACY_FASTPATH=0',))
            cuda(LEGACY / 'scripts/replay_sts_full.cu', LEGACY / 'bin/replay_sts_full')
        elif component == 'corrected_dcf':
            cuda(CORRECTED / 'scripts/replay_corrected_dcf_full.cu', CORRECTED / 'bin/replay_corrected_dcf_full', defines=('LEGACY_FASTPATH=1',))
            cuda(CORRECTED / 'scripts/replay_sts_export.cu', CORRECTED / 'bin/replay_sts_export')
            cuda(CORRECTED / 'scripts/zero_v_control.cu', CORRECTED / 'bin/zero_v_control',
                 defines=('OP_RECON_SKIP_ZERO_WEIGHTED_VISS=1', 'OP_RECON_DEBUG_CHECK_FINITE=1'), crypto=False)
        elif component == 'cpu_nufft':
            for name in ('T05_20261101_matched', 'T22_20261101_masked_roi'):
                task = EXCHANGE / '05_baselines' / name / 'scripts'
                execute([args.host_cxx, '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp',
                         task / 'direct.cpp', '-o', task / 'direct.so'], task / 'direct.so')
        elif component == 'microchunk':
            task = EXCHANGE / '09_memory_boundary/T09_20261101_production_microchunk_regression'
            cuda(task / 'source/main_production_regression.cu',
                 task / 'bin/production_microchunk_regression', (task / 'source',))
        elif component == 'mpi':
            execute(['bash', ROOT / 'cluster_mpi/scripts/build.sh', '--archs', args.arch[3:], '--nvcc', args.nvcc])
    receipt = {'schema': 'public_artifact_build_v1', 'state': 'dry_run' if args.dry_run else 'compiled',
               'components': list(selected), 'arch': args.arch, 'commands': commands,
               'gpu_experiments_started': False,
               'outputs': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p), 'bytes': p.stat().st_size}
                           for p in outputs if p.is_file() and not args.dry_run]}
    if not args.dry_run:
        directory = ROOT / 'build'
        directory.mkdir(exist_ok=True)
        (directory / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'state': receipt['state'], 'components': list(selected), 'gpu_experiments_started': False}))


if __name__ == '__main__':
    main()
