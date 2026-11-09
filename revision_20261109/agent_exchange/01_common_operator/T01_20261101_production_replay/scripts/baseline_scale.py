#!/usr/bin/env python3
"""Complete cached single-day baseline ranges; CPU only, no new geometry."""
import argparse
import json
from pathlib import Path
import numpy as np

FIELDS=('u','v','w','x1','y1','z1','inv1','x2','y2','z2','inv2','raw_re','raw_im','re','im')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    dtype=np.dtype([(field,'<f4') for field in FIELDS]);days=[];quantiles=[0,.01,.05,.25,.5,.75,.95,.99,1]
    audit=json.loads((root/'artifact_audit.json').read_text());hashes={a['path']:a['sha256'] for a in audit['cache_artifacts']}
    for day in (1,113,225,337,450):
        length=[];sources=[]
        for seg in range(1,11):
            path=root/'cache'/f'day{day}_seg{seg}.bin'
            rows=np.memmap(path,dtype=dtype,mode='r',offset=36)
            u=rows['u'].astype('f8');v=rows['v'].astype('f8');w=rows['w'].astype('f8')
            length.append(np.sqrt(u*u+v*v+w*w));sources.append({'path':str(path),'sha256':hashes[str(path)]})
        values=np.concatenate(length);q=np.quantile(values,quantiles)
        days.append({'day':day,'half_baseline_records':len(values),'implied_full_signed_records':2*len(values),
            'quantiles':quantiles,'length_lambda':q.tolist(),'length_m':(30*q).tolist(),
            'lambda_over_Bmax_rad':float(1/q[-1]),'lambda_over_Bmax_arcmin':float(180*60/np.pi/q[-1]),
            'sources':sources})
    angular=np.sqrt(4*np.pi/(12*4096**2))
    result={'days':days,'frequency_hz':1e7,'lambda_m':30,'nside':4096,'NEST_pixel_count':12*4096**2,
        'characteristic_equal_area_pixel_angle':{'rad':float(angular),'deg':float(angular*180/np.pi),'arcmin':float(angular*180*60/np.pi)},
        'scope':'all1954680 cached production half-baselines on each of five discrete orbital days; not all450 days, not continuous orbit extrema',
        'interpretation':'lambda/Bmax is a nominal angular scale, not measured dirty-beam FWHM, not a Nyquist proof or scientific requirement for Nside4096. Pixel angle is sqrt(equal pixel area), not maximum/minimum HEALPix pixel width.'}
    (root/'baseline_scale.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pixel_arcmin':result['characteristic_equal_area_pixel_angle']['arcmin'],'days':[{'day':r['day'],'min_m':r['length_m'][0],'max_m':r['length_m'][-1],'lambda_over_Bmax_arcmin':r['lambda_over_Bmax_arcmin']} for r in days]},indent=2))
if __name__=='__main__':main()
