#!/usr/bin/env python3
"""Regenerate and pool corrected direct N and W on a fixed, isolated GPU lane."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager

import numpy as np

NSIDE = 4096
PIXELS = 12 * NSIDE**2
SEGMENTS = 10
TIMESTAMPS = 6981
RECORDS_PER_SEGMENT = 195468
CACHE_HEADER = struct.Struct('<8s7i')
CACHE_BYTES = CACHE_HEADER.size + RECORDS_PER_SEGMENT * 15 * 4
GIB = 1024**3
BLOCK = 4 * 1024**2
DIRECT_FILES = {'N': ('raw_N.f32', '<f4', 'rawN'), 'W': ('W.f64', '<f8', 'W')}
PROTOCOL = {
    'schema': 'direct450_common_protocol_v1',
    'frequency_hz': 10000000,
    'nside': NSIDE,
    'ordering': 'NEST',
    'pixels': PIXELS,
    'observation_days': 450,
    'sky': {'bytes': 805306368, 'sha256': '277b6cb7299f2e8e54cc1758a045e7589686e28dc79a1df2f56aba5fc2d2d5ac'},
    'dcf': {'bytes': 80064, 'sha256': '8f5d7f206e5dc1d51282b5be43bfbc237b61d7d63a4665ee10997a244c7e4d2f'},
    'forward': {'method': 'production_replay_cache_only', 'orbit_seed': 42, 'gpus': 1,
                'blockage': 1, 'occ_mode': 2, 'viss_tile_pix': 256, 'grid_cell_scale': 1,
                'gen_gpu_index': 0, 'time_start': 0, 'time_count': 0, 'profile': 0,
                'segments': SEGMENTS, 'timestamps_per_segment': TIMESTAMPS,
                'half_baselines_per_segment': RECORDS_PER_SEGMENT,
                'input_cache': 'PREPLAY1 regenerated per day on the lane GPU',
                'visibility': 'consumed phase-corrected half-set complex float32'},
    'inverse': {'method': 'full_corrected_dcf_weighted_reference', 'daily_N': '<f4',
                'daily_W': '<f8', 'lane_N': '<f8', 'lane_W': '<f8',
                'normalization': 'divide only after merging all N and W sums'},
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def sha_file(path):
    with path.open('rb') as handle:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(handle, 'sha256').hexdigest()
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(BLOCK), b''):
            digest.update(block)
        return digest.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_dates(value):
    days = []
    for term in value.split(','):
        match = re.fullmatch(r'(\d+)(?:-(\d+)(?::(\d+))?)?', term)
        if match is None:
            raise argparse.ArgumentTypeError('use explicit days or ranges such as 1,3-449:2')
        first = int(match[1])
        last = int(match[2] or first)
        stride = int(match[3] or 1)
        if not 1 <= first <= last <= 450 or stride < 1:
            raise argparse.ArgumentTypeError('days must be in 1..450 with a positive stride')
        days.extend(range(first, last + 1, stride))
    if len(days) != len(set(days)):
        raise argparse.ArgumentTypeError('duplicate assigned day')
    return sorted(days)


def file_stat(path):
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), f'expected a regular file: {path}')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def frozen_file(path, expected=None, executable=False):
    initial = file_stat(path)
    require(not executable or os.access(path, os.X_OK), f'not executable: {path}')
    identity = {'path': str(path), 'bytes': initial[2], 'sha256': sha_file(path)}
    require(file_stat(path) == initial, f'file changed while hashing: {path}')
    if expected is not None:
        require(all(identity[key] == value for key, value in expected.items()), f'frozen identity mismatch: {path}')
    return identity, initial


def check_inputs(snapshots):
    for path, initial in snapshots.items():
        require(file_stat(path) == initial, f'input changed during lane: {path}')


@contextmanager
def lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'active lane or GPU lock: {path}') from exc
        yield fd
    finally:
        os.close(fd)


def nvidia_query(*query):
    completed = subprocess.run(['nvidia-smi', *query, '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True, timeout=20)
    return [tuple(value.strip() for value in line.split(','))
            for line in completed.stdout.strip().splitlines()]


def gpu_contexts(gpu_uuid):
    contexts = nvidia_query('--query-compute-apps=gpu_uuid,pid')
    require(all(len(row) == 2 and re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', row[0])
                and row[1].isdigit() for row in contexts), 'unrecognized compute-context inventory')
    return [int(pid) for uid, pid in contexts if uid.lower() == gpu_uuid.lower()]


def gpu_boundary(args):
    rows = nvidia_query('-i', args.gpu_uuid,
                        '--query-gpu=uuid,name,memory.total,memory.free,driver_version,pci.bus_id')
    require(len(rows) == 1 and len(rows[0]) == 6 and rows[0][0].lower() == args.gpu_uuid.lower(),
            'GPU inventory differs from the selected UUID')
    uid, name, total, free, driver, bus = rows[0]
    contexts = gpu_contexts(uid)
    require(not contexts, f'target GPU {uid} has compute contexts {contexts}')
    require(int(free) >= args.min_gpu_memory_mib, f'target GPU {uid} has only {free} MiB free')
    return {'uuid': uid, 'name': name, 'memory_total_mib': int(total), 'memory_free_mib': int(free),
            'driver_version': driver, 'pci_bus_id': bus}


def guard_disk(root, minimum_gib):
    free = shutil.disk_usage(root).free
    require(free >= minimum_gib * GIB, f'only {free / GIB:.2f} GiB free on {root}')
    return free


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except ProcessLookupError:
        process.wait()
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def invoke(command, folder, args, lock_fds):
    gpu = gpu_boundary(args)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS=str(args.omp_threads), LC_ALL='C')
    invocation = {'command': command, 'started_at': now(), 'gpu': gpu,
                  'cwd': str(folder), 'omp_threads': args.omp_threads}
    atomic_json(folder / 'invocation.json', invocation)
    begin = time.monotonic()
    with (folder / 'stdout.log').open('xb') as output:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True, pass_fds=lock_fds)
        try:
            while True:
                try:
                    status = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    require(time.monotonic() - begin < args.stage_timeout_s, 'GPU stage timed out')
                    foreign = [pid for pid in gpu_contexts(args.gpu_uuid) if pid != process.pid]
                    require(not foreign, f'foreign target-GPU contexts during stage: {foreign}')
        finally:
            if process.poll() is None:
                terminate(process)
    invocation.update(elapsed_s=time.monotonic() - begin, returncode=status, finished_at=now())
    atomic_json(folder / 'invocation.json', invocation)
    require(status == 0, f'GPU stage exited {status}: {folder / "stdout.log"}')
    invocation['gpu_after'] = gpu_boundary(args)
    return invocation


def validate_cache(cache_root, day, forward_log):
    expected_names = {f'day{day}_seg{segment}.bin' for segment in range(1, SEGMENTS + 1)}
    require({path.name for path in cache_root.glob('*.bin')} == expected_names,
            'cache file set differs from the ten expected segments')
    log_text = forward_log.read_text(encoding='utf-8', errors='replace')
    require(re.search(r'^Visible logical GPUs: 0\s*$', log_text, re.M) is not None,
            'forward log did not confirm one visible logical GPU')
    marks = re.findall(r'^CACHE_ONLY_DONE day=(\d+) seg=(\d+)/(\d+)\s*$', log_text, re.M)
    require(marks == [(str(day), str(segment), str(SEGMENTS)) for segment in range(1, SEGMENTS + 1)],
            'forward log did not complete all requested cache segments')
    vis = hashlib.sha256()
    geometry = hashlib.sha256()
    segment_receipts = []
    for segment in range(1, SEGMENTS + 1):
        path = cache_root / f'day{day}_seg{segment}.bin'
        content = path.read_bytes()
        require(len(content) == CACHE_BYTES, f'cache byte count mismatch: {path}')
        expected = (b'PREPLAY1', NSIDE, day, segment, TIMESTAMPS * (segment - 1),
                    TIMESTAMPS, RECORDS_PER_SEGMENT, 1)
        require(CACHE_HEADER.unpack_from(content) == expected, f'cache header mismatch: {path}')
        rows = np.frombuffer(content, dtype='<f4', offset=CACHE_HEADER.size).reshape(-1, 15)
        require(np.isfinite(rows).all(), f'nonfinite cache records: {path}')
        geom_bytes = rows[:, :11].tobytes(order='C')
        vis_bytes = rows[:, 13:15].tobytes(order='C')
        geometry.update(geom_bytes)
        vis.update(vis_bytes)
        segment_receipts.append({'segment': segment, 'name': path.name, 'bytes': len(content),
                                 'records': len(rows), 'sha256': hashlib.sha256(content).hexdigest(),
                                 'geometry_sha256': hashlib.sha256(geom_bytes).hexdigest(),
                                 'visibility_sha256': hashlib.sha256(vis_bytes).hexdigest()})
    return {'records': SEGMENTS * RECORDS_PER_SEGMENT, 'segments': SEGMENTS,
            'segment_receipts': segment_receipts,
            'geometry_sha256': geometry.hexdigest(), 'visibility_sha256': vis.hexdigest()}


def require_sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            f'invalid SHA-256 for {label}')


def check_array(path, entry, dtype, pixels):
    require(entry.get('format', entry.get('dtype')) == dtype and entry.get('count') == pixels
            and entry.get('bytes') == pixels * np.dtype(dtype).itemsize,
            f'array metadata mismatch: {path}')
    require(file_stat(path)[2] == entry['bytes'], f'array byte count mismatch: {path}')
    require_sha(entry.get('sha256'), path)


def direct_outputs(folder, day, identities):
    metrics = read_json(folder / 'metrics.json')
    expected = {'state': 'completed', 'day': day, 'nside': NSIDE, 'segments': SEGMENTS,
                'half_baselines': identities['records'], 'output_pixels': PIXELS,
                'method': PROTOCOL['inverse']['method'], 'ordering': PROTOCOL['ordering'],
                'forward_executed': False, 'forward_gpu_count': 1,
                'dcf_sha256': PROTOCOL['dcf']['sha256'], 'dcf_days': PROTOCOL['observation_days'],
                'orbit_seed': PROTOCOL['forward']['orbit_seed']}
    require(all(metrics.get(key) == value for key, value in expected.items()),
            f'incomplete or mismatched direct metrics: {folder}')
    inverse_s = metrics.get('inverse_loop_wall_s')
    require(isinstance(inverse_s, (int, float)) and math.isfinite(inverse_s) and inverse_s > 0,
            'invalid direct inverse timer')
    cache_ids = [{key: entry[key] for key in ('segment', 'name', 'bytes', 'sha256')}
                 for entry in identities['segment_receipts']]
    require(metrics.get('caches') == cache_ids, 'inverse cache identities differ from verified forward caches')
    stats = read_json(folder / 'full_stats.json')
    for name, dtype, stats_key in DIRECT_FILES.values():
        entry = stats[stats_key]
        check_array(folder / name, entry, dtype, PIXELS)
        require(entry.get('file') == name and entry.get('finite_count') == PIXELS
                and entry.get('nonfinite') == 0, f'invalid direct output summary: {name}')
    require(0 <= stats['W']['min'] <= stats['W']['max'] and stats['W']['max'] > 0,
            'direct output weight extrema invalid')
    return metrics, stats


def pool_array(name, daily_path, daily, prior_path, prior, output, pixels):
    daily_sha = hashlib.sha256()
    prior_sha = hashlib.sha256()
    output_sha = hashlib.sha256()
    daily_dtype = np.dtype(daily.get('format', daily['dtype']))
    minimum = math.inf
    maximum = -math.inf
    with daily_path.open('rb') as daily_file, output.open('xb') as out_file:
        prior_file = prior_path.open('rb') if prior_path is not None else None
        try:
            for start in range(0, pixels, BLOCK // 4):
                count = min(BLOCK // 4, pixels - start)
                day_bytes = daily_file.read(count * daily_dtype.itemsize)
                require(len(day_bytes) == count * daily_dtype.itemsize, f'truncated daily {name}')
                daily_sha.update(day_bytes)
                value = np.frombuffer(day_bytes, dtype=daily_dtype).astype('<f8')
                require(np.isfinite(value).all(), f'nonfinite daily {name}')
                if name == 'W':
                    require((value >= 0).all(), 'negative daily W')
                if prior_file is not None:
                    prior_bytes = prior_file.read(count * 8)
                    require(len(prior_bytes) == count * 8, f'truncated checkpoint {name}')
                    prior_sha.update(prior_bytes)
                    value += np.frombuffer(prior_bytes, dtype='<f8')
                    require(np.isfinite(value).all(), f'nonfinite pooled {name}')
                minimum = min(minimum, float(value.min()))
                maximum = max(maximum, float(value.max()))
                chunk = value.tobytes()
                out_file.write(chunk)
                output_sha.update(chunk)
            require(not daily_file.read(1) and daily_sha.hexdigest() == daily['sha256'],
                    f'daily {name} SHA-256 mismatch')
            if prior_file is not None:
                require(not prior_file.read(1) and prior_sha.hexdigest() == prior['sha256'],
                        f'prior checkpoint {name} SHA-256 mismatch')
            out_file.flush()
            os.fsync(out_file.fileno())
        finally:
            if prior_file is not None:
                prior_file.close()
    return {'path': output.name, 'bytes': 8 * pixels, 'sha256': output_sha.hexdigest(),
            'dtype': '<f8', 'count': pixels, 'min': minimum, 'max': maximum}


def load_checkpoint(root, lane):
    path = root / 'CURRENT.json'
    if not path.exists():
        return None, None
    current = read_json(path)
    require(current.get('schema') == 'direct450_lane_current_v1', 'invalid CURRENT schema')
    relative = current['manifest_path']
    require(re.fullmatch(r'generations/g\d{3}_d\d{3}_[0-9a-f]{12}/manifest\.json', relative) is not None,
            'invalid CURRENT manifest path')
    manifest_path = root / relative
    require(sha_file(manifest_path) == current['manifest_sha256'], 'CURRENT manifest hash mismatch')
    manifest = read_json(manifest_path)
    expected = {'schema': 'direct450_lane_generation_v1', 'status': 'committed',
                'protocol': PROTOCOL, 'protocol_sha256': canonical_sha(PROTOCOL),
                'lane': lane, 'assigned_days': lane['assigned_days']}
    require(all(manifest.get(key) == value for key, value in expected.items()), 'checkpoint identity mismatch')
    days = manifest['completed_days']
    require(days and days == current['completed_days'] == lane['assigned_days'][:len(days)]
            and days == [entry['day'] for entry in manifest['days']], 'checkpoint is not an assigned prefix')
    require(all(entry.get('status') == 'accepted' for entry in manifest['days']), 'unaccepted checkpoint day')
    require(set(manifest['arrays']) == {'N', 'W'}, 'checkpoint must contain both N and W')
    for name, entry in manifest['arrays'].items():
        require(entry['path'] == f'{name}.f64', 'invalid checkpoint array path')
        array = manifest_path.parent / entry['path']
        check_array(array, entry, '<f8', PIXELS)
        require(sha_file(array) == entry['sha256'], f'checkpoint array hash mismatch: {array}')
    return manifest_path, manifest


def archive_attempt(root, attempt, state, detail):
    archive = root / 'receipts' / attempt.name
    archive.mkdir(exist_ok=True)
    for stage, filenames in {
        'forward': ('stdout.log', 'invocation.json'),
        'direct': ('stdout.log', 'invocation.json', 'metrics.json', 'full_stats.json', 'segments.csv'),
    }.items():
        for name in filenames:
            source = attempt / stage / name
            if source.is_file():
                destination = archive / f'{stage}_{name}'
                with source.open('rb') as original, destination.open('wb') as output:
                    shutil.copyfileobj(original, output)
                    output.flush()
                    os.fsync(output.fileno())
        if stage == 'forward':
            for source in (attempt / stage).glob('profile_day*_10M.csv'):
                destination = archive / f'forward_{source.name}'
                with source.open('rb') as original, destination.open('wb') as output:
                    shutil.copyfileobj(original, output)
                    output.flush()
                    os.fsync(output.fileno())
    atomic_json(archive / 'status.json', {'state': state, 'at': now(), 'detail': detail})
    sync_dir(archive.parent)
    return archive


def recover(root, current_path, manifest):
    completed = set(manifest['completed_days']) if manifest is not None else set()
    for path in sorted((root / 'work').iterdir()):
        match = re.fullmatch(r'd(\d{3})_[0-9a-f]{12}', path.name)
        if match and path.is_dir() and not path.is_symlink():
            state = 'committed' if int(match[1]) in completed else 'interrupted'
            archive_attempt(root, path, state, 'recovered after an earlier invocation')
            shutil.rmtree(path)
    # A generation with no CURRENT reference has no authority. Keep only the verified one.
    current_dir = current_path.parent if current_path is not None else None
    for path in (root / 'generations').iterdir():
        if (re.fullmatch(r'g\d{3}_d\d{3}_[0-9a-f]{12}', path.name)
                and path != current_dir and path.is_dir() and not path.is_symlink()):
            shutil.rmtree(path)
    sync_dir(root / 'work')
    sync_dir(root / 'generations')


def run_day(root, args, lane, snapshots, previous_path, previous, day, lock_fds):
    attempt_id = uuid.uuid4().hex[:12]
    attempt = root / 'work' / f'd{day:03d}_{attempt_id}'
    forward = attempt / 'forward'
    direct = attempt / 'direct'
    cache = attempt / 'cache'
    for folder in (forward, direct, cache):
        folder.mkdir(parents=True)
    begin = time.monotonic()
    started_at = now()
    print(f'{started_at} day={day} state=started attempt={attempt.name}', flush=True)
    try:
        check_inputs(snapshots)
        forward_run = invoke([
            str(args.forward_bin), f'--gpus={args.gpu_uuid}', '--btag=10M', f'--nside={NSIDE}',
            f'--day_start={day}', '--day_count=1', f'--segs={SEGMENTS}', '--occ_mode=2', '--blockage=1',
            '--orbit_seed=42', '--gen_gpu_index=0', '--time_start=0', '--time_count=0',
            '--grid_cell_scale=1', '--B_mode=bin', '--C_mode=bin',
            f'--sky_dir={args.sky_file.parent}', f'--out_dir={forward}', f'--cache_root={cache}',
            '--cache_only=1', '--write_output=0', '--write_viss=0', '--write_baseline_txt=0',
            '--profile=0', '--viss_tile_pix=256',
        ], forward, args, lock_fds)
        identities = validate_cache(cache, day, forward / 'stdout.log')
        check_inputs(snapshots)
        direct_run = invoke([
            str(args.direct_bin), f'--gpus={args.gpu_uuid}', f'--cache_root={cache}',
            f'--out_dir={direct}', f'--dcf_bin={args.dcf_bin}', f'--nside={NSIDE}',
            f'--day={day}', f'--max_segments={SEGMENTS}',
        ], direct, args, lock_fds)
        metrics, stats = direct_outputs(direct, day, identities)
        verified_cache = validate_cache(cache, day, forward / 'stdout.log')
        require(verified_cache == identities, 'visibility cache changed during inverse')
        check_inputs(snapshots)
        archive = archive_attempt(root, attempt, 'validated_daily_outputs', {'day': day})
        completed = ([] if previous is None else previous['completed_days']) + [day]
        generation = f'g{len(completed):03d}_d{day:03d}_{attempt_id}'
        pending = attempt / 'generation'
        pending.mkdir()
        arrays = {}
        pool_start = time.monotonic()
        for label, (filename, _, stats_key) in DIRECT_FILES.items():
            prior = None if previous is None else previous['arrays'][label]
            prior_path = None if prior is None else previous_path.parent / prior['path']
            arrays[label] = pool_array(label, direct / filename, stats[stats_key], prior_path,
                                       prior, pending / f'{label}.f64', PIXELS)
        require(arrays['W']['min'] >= 0 and arrays['W']['max'] > 0, 'pooled W extrema invalid')
        check_inputs(snapshots)
        receipt = {'day': day, 'status': 'accepted', 'started_at': started_at, 'verified_at': now(),
                   'diagnostics_path': str(archive.relative_to(root)),
                   'forward': forward_run, 'direct': direct_run,
                   'direct_inverse_loop_wall_s': metrics['inverse_loop_wall_s'],
                   'direct_arrays': stats, 'checkpoint_pool_s': time.monotonic() - pool_start,
                   'elapsed_to_commit_s': time.monotonic() - begin, **identities}
        manifest = {'schema': 'direct450_lane_generation_v1', 'status': 'committed',
                    'created_at': now(), 'protocol': PROTOCOL, 'protocol_sha256': canonical_sha(PROTOCOL),
                    'lane': lane, 'assigned_days': lane['assigned_days'], 'completed_days': completed,
                    'arrays': arrays, 'days': ([] if previous is None else previous['days']) + [receipt]}
        atomic_json(pending / 'manifest.json', manifest)
        destination = root / 'generations' / generation
        pending.rename(destination)
        sync_dir(destination.parent)
        sync_dir(attempt)
        manifest_path = destination / 'manifest.json'
        atomic_json(root / 'CURRENT.json', {'schema': 'direct450_lane_current_v1', 'generation': generation,
                                           'manifest_path': str(manifest_path.relative_to(root)),
                                           'manifest_sha256': sha_file(manifest_path),
                                           'completed_days': completed})
        atomic_json(archive / 'status.json', {'state': 'committed', 'at': now(), 'day': day,
                                              'visibility_sha256': identities['visibility_sha256'],
                                              'records': identities['records'], 'elapsed_s': time.monotonic() - begin})
        print(f'{now()} day={day} state=committed elapsed_s={time.monotonic()-begin:.3f} '
              f'records={identities["records"]} visibility_sha256={identities["visibility_sha256"]}', flush=True)
        shutil.rmtree(attempt)
        sync_dir(attempt.parent)
        if previous_path is not None:
            shutil.rmtree(previous_path.parent)
            sync_dir(destination.parent)
        return manifest_path, manifest
    except BaseException as exc:
        if attempt.exists():
            current = read_json(root / 'CURRENT.json') if (root / 'CURRENT.json').exists() else {}
            state = 'committed' if day in current.get('completed_days', []) else 'failed'
            archive_attempt(root, attempt, state, {'day': day, 'error': str(exc),
                                                   'elapsed_s': time.monotonic() - begin})
            shutil.rmtree(attempt)
            sync_dir(attempt.parent)
        raise


def run(args):
    root = args.lane_root
    root.mkdir(parents=True, exist_ok=True)
    gpu_lock_dir = root.parent / '.gpu_locks'
    gpu_lock_dir.mkdir(exist_ok=True)
    require(gpu_lock_dir.is_dir() and not gpu_lock_dir.is_symlink(), f'unsafe GPU lock directory: {gpu_lock_dir}')
    gpu_lock = gpu_lock_dir / f'{args.gpu_uuid.lower()}.lock'
    with lock(root / '.lane.lock') as lane_fd, lock(gpu_lock) as gpu_fd:
        gpu = gpu_boundary(args)
        free = guard_disk(root, args.min_free_gib)
        require(args.sky_file.name == 'B_10M.bin', 'forward executable requires the basename B_10M.bin')
        paths = {'sky': args.sky_file, 'dcf': args.dcf_bin, 'forward_bin': args.forward_bin,
                 'direct_bin': args.direct_bin}
        if args.build_receipt is not None:
            paths['build_receipt'] = args.build_receipt
        identities = {}
        snapshots = {}
        for key, path in paths.items():
            identities[key], snapshots[path] = frozen_file(path, PROTOCOL.get(key), executable=key.endswith('_bin') and key != 'dcf')
        lane = {'lane_id': root.name, 'host': os.uname().nodename,
                'gpu': {key: value for key, value in gpu.items() if key != 'memory_free_mib'},
                'assigned_days': args.dates, 'inputs': identities, 'omp_threads': args.omp_threads,
                'runner_sha256': sha_file(Path(__file__).resolve())}
        freeze = {'schema': 'direct450_lane_freeze_v1', 'protocol': PROTOCOL,
                  'protocol_sha256': canonical_sha(PROTOCOL), 'lane': lane}
        freeze_path = root / 'lane.json'
        if freeze_path.exists():
            require(read_json(freeze_path) == freeze, 'lane configuration/input identity changed')
        else:
            require(set(path.name for path in root.iterdir()) == {'.lane.lock'},
                    'new lane requires an empty, dedicated lane directory')
            atomic_json(freeze_path, freeze)
        for folder in ('work', 'generations', 'receipts'):
            path = root / folder
            path.mkdir(exist_ok=True)
            require(path.is_dir() and not path.is_symlink(), f'unsafe lane directory: {path}')
        checkpoint_path, checkpoint = load_checkpoint(root, lane)
        completed = [] if checkpoint is None else checkpoint['completed_days']
        if args.phase == 'preflight':
            print(json.dumps({'status': 'preflight_ok', 'at': now(), 'lane': lane,
                              'gpu_free_mib': gpu['memory_free_mib'], 'disk_free_bytes': free,
                              'completed_days': completed, 'protocol_sha256': freeze['protocol_sha256']}, indent=2))
            return
        recover(root, checkpoint_path, checkpoint)
        try:
            for day in args.dates[len(completed):]:
                require(not (root / 'PAUSE').exists(), 'PAUSE marker present at day boundary')
                guard_disk(root, args.min_free_gib)
                atomic_json(root / 'STATUS.json', {'state': 'running', 'day': day, 'at': now(),
                                                  'completed_days': completed})
                checkpoint_path, checkpoint = run_day(root, args, lane, snapshots, checkpoint_path,
                                                       checkpoint, day, (lane_fd, gpu_fd))
                completed = checkpoint['completed_days']
        except BaseException as exc:
            current = read_json(root / 'CURRENT.json') if (root / 'CURRENT.json').exists() else {}
            atomic_json(root / 'STATUS.json', {'state': 'stopped', 'day': day, 'at': now(),
                                              'completed_days': current.get('completed_days', []),
                                              'error': str(exc)})
            raise
        atomic_json(root / 'STATUS.json', {'state': 'complete', 'at': now(), 'completed_days': completed})
        print(f'{now()} state=complete days={len(completed)} current={root / "CURRENT.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'run'), required=True)
    parser.add_argument('--lane-root', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--dates', required=True, type=parse_dates)
    for name in ('forward-bin', 'direct-bin', 'sky-file', 'dcf-bin'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--build-receipt', type=Path)
    parser.add_argument('--omp-threads', type=int, default=8)
    parser.add_argument('--min-gpu-memory-mib', type=int, default=8500)
    parser.add_argument('--min-free-gib', type=int, default=12)
    parser.add_argument('--stage-timeout-s', type=int, default=7200)
    args = parser.parse_args()
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', args.gpu_uuid):
        parser.error('--gpu-uuid must be exactly one complete physical GPU UUID')
    if min(args.omp_threads, args.min_gpu_memory_mib, args.min_free_gib, args.stage_timeout_s) < 1:
        parser.error('thread, memory, disk, and timeout limits must be positive')
    for name in ('forward_bin', 'direct_bin', 'sky_file', 'dcf_bin', 'build_receipt', 'lane_root'):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    args.gpu_uuid = 'GPU-' + args.gpu_uuid[4:].lower()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        run(args)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        print(f'{now()} state=stopped {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


def stop(signum, _frame):
    raise RuntimeError(f'signal {signum} received')


if __name__ == '__main__':
    sys.exit(main())
