#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    base=args.root/'cpu_t11';unit=json.loads((base/'metrics_unit.json').read_text());legacy=json.loads((base/'metrics_legacy.json').read_text())
    u=np.load(base/'beams_unit.npz');d=np.load(base/'beams_legacy.npz')
    assert np.array_equal(u['source_pixels'],d['source_pixels'])
    rows=[]
    for k,(a,b) in enumerate(zip(unit['sources'],legacy['sources'])):
        ref=u['beams_norm'][k];out=d['beams_norm'][k];valid=u['valid']&d['valid']
        rows.append({'source':a['source'],'pixel':a['pixel_index'],
            'visible_fraction':a['variant_metrics']['visible_fraction'],
            'natural_diagonal_peak':a['direct_peak'],'legacy_dcf_diagonal_peak':b['direct_peak'],
            'same_source_normalized_legacy_vs_unit_relative_l2':float(np.linalg.norm(out[valid]-ref[valid])/np.linalg.norm(ref[valid])),
            'same_source_normalized_legacy_vs_unit_pearson':float(np.corrcoef(ref[valid],out[valid])[0,1]),
            'unit_plane_shape':a['variant_metrics']['plane_raw'].get('shape_vs_direct_normalized'),
            'unit_representative_shape':a['variant_metrics']['midpoint_normalized'].get('shape_vs_direct_normalized'),
            'legacy_representative_shape':b['variant_metrics']['midpoint_normalized'].get('shape_vs_direct_normalized')})
    result={'sources':rows,'phase':'consumed n convention in exact forward and backprojection, matching production host phase correction',
        'supported':'Same-source direct-vs-STS and unit-vs-legacy-DCF response differences; source-dependent diagonal amplitude/visible count.',
        'not_supported':'Different-source response maps have not been recentered/rotated into a common tangent frame. Their pair L2 does not prove shape shift-variance. No sky recovery claim.',
        'scope':'CPU complex128 1120x2048 fixture, W=I control and immutable historical DCF diagnostic; not CUDA STS full-sky replay.'}
    (base/'same_source_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
