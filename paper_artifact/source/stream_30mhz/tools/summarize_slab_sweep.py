#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def f(row, key):
    return float(row[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--csv-out', required=True)
    ap.add_argument('--md-out', required=True)
    args = ap.parse_args()

    with open(args.input, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit('No rows found in summary CSV')

    groups = defaultdict(list)
    for row in rows:
        groups[int(row['slab_pix'])].append(row)

    fields = [
        'slab_pix', 'slab_gpix', 'slabs_per_gpu_max', 'slabs_total', 'repeats',
        'visibility_mean_s', 'visibility_std_s',
        'reconstruction_mean_s', 'reconstruction_std_s',
        'segment_loop_mean_s', 'segment_loop_std_s',
        'total_day_mean_s', 'total_day_std_s',
        'peak_gpu_mem_mean_gib', 'peak_gpu_mem_max_gib',
        'occupied_groups_mean'
    ]
    out = []
    for slab in sorted(groups):
        rr = groups[slab]
        def vals(k): return [f(x, k) for x in rr]
        def mean(k): return statistics.fmean(vals(k))
        def std(k):
            v = vals(k)
            return statistics.stdev(v) if len(v) > 1 else 0.0
        out.append({
            'slab_pix': slab,
            'slab_gpix': slab / 1e9,
            'slabs_per_gpu_max': max(int(x['slabs_per_gpu_max']) for x in rr),
            'slabs_total': max(int(x['slabs_total']) for x in rr),
            'repeats': len(rr),
            'visibility_mean_s': mean('visibility_s'),
            'visibility_std_s': std('visibility_s'),
            'reconstruction_mean_s': mean('reconstruction_s'),
            'reconstruction_std_s': std('reconstruction_s'),
            'segment_loop_mean_s': mean('segment_loop_s'),
            'segment_loop_std_s': std('segment_loop_s'),
            'total_day_mean_s': mean('total_day_s'),
            'total_day_std_s': std('total_day_s'),
            'peak_gpu_mem_mean_gib': mean('peak_gpu_mem_gib'),
            'peak_gpu_mem_max_gib': max(vals('peak_gpu_mem_gib')),
            'occupied_groups_mean': mean('occupied_groups_total'),
        })

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open('w') as fh:
        fh.write('| Slab pixels | Slabs/GPU | Total slabs | Visibility (s) | Reconstruction (s) | Segment loop (s) | Total day (s) | Peak GPU memory (GiB) |\n')
        fh.write('|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in out:
            fh.write(
                f"| {r['slab_pix']:,} | {r['slabs_per_gpu_max']} | {r['slabs_total']} | "
                f"{r['visibility_mean_s']:.2f} ± {r['visibility_std_s']:.2f} | "
                f"{r['reconstruction_mean_s']:.2f} ± {r['reconstruction_std_s']:.2f} | "
                f"{r['segment_loop_mean_s']:.2f} ± {r['segment_loop_std_s']:.2f} | "
                f"{r['total_day_mean_s']:.2f} ± {r['total_day_std_s']:.2f} | "
                f"{r['peak_gpu_mem_max_gib']:.2f} |\n"
            )

    print(f'Wrote {csv_path}')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()
