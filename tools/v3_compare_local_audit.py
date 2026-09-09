"""CPU structural audit of saved V3 predictions with explicit case identifiers.

Rules are applied identically to all arms and RL3 controls. Native depth samples
are never interpolated for axial separation. GT is read only by this evaluator.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
from scipy.ndimage import label, center_of_mass, map_coordinates
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from tools.v3_compare_evaluation import all_cases, csv_write, _structure_row, _um_slice
from tools import v3_compare_experiment as exp
from datasets.matlab_multivolume_dataset import load_inference_input, DatasetItemKey, _read_targets

Z = np.arange(10., 101., 10.)
PITCH = 220 / 4 / 49
THRESHOLDS = (.05, .10, .20)

def peaks(profile):
    """Collapse a flat top to one peak; a zero profile contains no peaks."""
    if np.max(profile) <= 0:
        return np.array([], dtype=int)
    return find_peaks(np.pad(profile, (1, 1), constant_values=-1e-30), plateau_size=1)[0] - 1

def match(expected, candidates, tolerance):
    if len(candidates) == 0:
        return {}
    cost = np.abs(np.asarray(expected)[:, None] - np.asarray(candidates)[None, :])
    cost[cost > tolerance] = 1e6
    a, b = linear_sum_assignment(cost)
    return {int(i): int(candidates[j]) for i,j in zip(a,b) if cost[i,j] < 1e6}

def distribution(pred, truth):
    p = pred / max(float(pred.sum()), 1e-30)
    t = truth / max(float(truth.sum()), 1e-30)
    occupied = t > 1e-6
    permitted = occupied.copy()
    permitted[:-1] |= occupied[1:]
    permitted[1:] |= occupied[:-1]
    return dict(local_depth_w1_um=float(np.abs(np.cumsum(p)-np.cumsum(t)).sum()*10),
                tail_mass=float(p[~permitted].sum()), peak_depth_um=float(Z[np.argmax(p)]),
                local_mass=float(pred.sum()))

def t03(pred, truth, metadata):
    groups = json.loads((exp.DATA/'T03/geometry.json').read_text())['geometry']
    rows, profiles = [], []
    threshold_base = float(pred.max())
    for index,g in enumerate(groups,1):
        x0,y0,x1,y1 = np.array(g['bbox_xy_one_based']) - 1
        bars = np.array(g['bars_xy_one_based']) - 1
        zprof = pred[:,y0:y1+1,x0:x1+1].sum((1,2))
        ztrue = truth[:,y0:y1+1,x0:x1+1].sum((1,2))
        zi = int(np.argmax(zprof))
        z_ok = abs(Z[zi]-g['depth_um']) <= 10
        image = pred[zi]
        horizontal = g['orientation']=='horizontal'
        start,end = (x0,x1) if horizontal else (y0,y1)
        # Integer pixel centers strictly inside central 60% of the line length.
        lo = int(np.ceil(start + .2*(end-start)))
        hi = int(np.floor(start + .8*(end-start)))
        profile = image[:,lo:hi+1].mean(1) if horizontal else image[lo:hi+1,:].mean(0)
        intervals = bars[:,[1,3]] if horizontal else bars[:,[0,2]]
        centers = intervals.mean(1)
        transverse = np.arange(max(0,int(intervals[0,0])-3),min(260,int(intervals[-1,1])+4))
        for position in transverse:
            profiles.append({**metadata,'diagnostic':'T03','region':index,'coordinate':int(position),
                             'coordinate_unit':'pixel','value':float(profile[position]),'selected_depth_um':float(Z[zi])})
        for threshold in THRESHOLDS:
            candidates = peaks(profile)
            candidates = candidates[(candidates>=transverse[0]) & (candidates<=transverse[-1])]
            candidates = candidates[profile[candidates]>=threshold*threshold_base]
            assignments = match(centers,candidates,2.)
            localized = len(assignments)==3 and z_ok
            ratios=[]
            for k in (0,1):
                if k in assignments and k+1 in assignments:
                    a,b=sorted((assignments[k],assignments[k+1]))
                    interior=profile[a+1:b]
                    ratios.append(float(interior.min()/max(min(profile[a],profile[b]),1e-30)) if interior.size else np.nan)
                else: ratios.append(np.nan)
            tracks=[image[a[1]:a[3]+1,lo:hi+1].max(0) if horizontal else image[lo:hi+1,a[0]:a[2]+1].max(1) for a in bars]
            rows.append({**metadata,'group':index,'orientation':g['orientation'],'width_px':g['width_px'],
                         'width_um':g['width_um'],'threshold_fraction':threshold,'all_three_localized':localized,
                         'matched_lines':len(assignments) if z_ok else 0,'missed_lines':3-len(assignments) if z_ok else 3,
                         'false_peak_count':len(candidates)-len(assignments),
                         'valley_ratio_1':ratios[0],'valley_ratio_2':ratios[1],
                         'separated':localized and all(np.isfinite(r) and r<=.8 for r in ratios),
                         'false_break_count':sum(bool((t<threshold*threshold_base).any()) for t in tracks),
                         **distribution(zprof,ztrue)})
    return rows,profiles

def t04(pred, truth, metadata):
    groups=json.loads((exp.DATA/'T04/geometry.json').read_text())['geometry']
    rows,profiles=[],[]
    for g in groups:
        ys,xs=_um_slice(g['roi_bounds_xy_um'],PITCH)
        profile=pred[:,ys,xs].sum((1,2))
        gtprof=truth[:,ys,xs].sum((1,2))
        peak_values=pred[:,ys,xs].max((1,2))
        expected=np.atleast_1d(g['z_um']).astype(float)
        for i,z in enumerate(Z):
            profiles.append({**metadata,'diagnostic':'T04','region':g['region_id'],
                             'coordinate':z,'coordinate_unit':'um','value':float(profile[i]),'gt_value':float(gtprof[i])})
        for threshold in THRESHOLDS:
            cand=peaks(profile)
            cand=cand[peak_values[cand]>=threshold*float(pred.max())]
            assignments=match(expected,Z[cand],10.)
            located=len(assignments)==len(expected)
            separation=g['separation_um'] if g['separation_um']!=[] else None
            applicable=len(expected)==2 and separation!=10
            ratio=np.nan
            separated='not_applicable'
            if applicable:
                separated=False
                if located:
                    a,b=sorted(int(np.argmin(abs(Z-v))) for v in assignments.values())
                    if b-a>=2:
                        ratio=float(profile[a+1:b].min()/max(min(profile[a],profile[b]),1e-30))
                        separated=ratio<=.8
            idx=np.array([np.argmin(abs(Z-z)) for z in expected])
            rows.append({**metadata,'region_id':g['region_id'],'family':g['family'],
                         'separation_um':separation,'threshold_fraction':threshold,
                         'all_layers_localized':located,'missed_layers':len(expected)-len(assignments),
                         'mean_z_error_layers':float(np.mean([abs(expected[k]-v)/10 for k,v in assignments.items()])) if assignments else np.nan,
                         'separation_applicable':applicable,'separated':separated,'valley_ratio':ratio,
                         'false_peak_count':len(cand)-len(assignments),
                         'expected_layer_energy_fraction':float(profile[idx].sum()/max(profile.sum(),1e-30)),
                         'deep_to_shallow_energy_ratio':float(profile[idx[-1]]/max(profile[idx[0]],1e-30)) if len(idx)==2 else np.nan,
                         **distribution(profile,gtprof)})
    return rows,profiles

def v03(pred, truth, metadata):
    beads=[g for g in json.loads((exp.DATA/'V03/geometry.json').read_text())['geometry'] if g['kind']=='solid_sphere']
    expected=np.array([[g['center_xyz_um'][2],g['center_xyz_um'][1]/PITCH,g['center_xyz_um'][0]/PITCH] for g in beads])
    rows, profiles=[],[]
    # Filled spheres have a broad, sometimes flat interior. Measure components'
    # intensity centroids; a single merged component can match only one bead.
    for threshold in THRESHOLDS:
        mask=(pred>=float(pred.max())*threshold) & (pred>0)
        labels,n=label(mask,structure=np.ones((3,3,3)))
        centers=np.array(center_of_mass(pred,labels,np.arange(1,n+1))) if n else np.empty((0,3))
        if n: centers[:,0]=10+10*centers[:,0]
        assignments={}
        if n:
            dz=np.abs(expected[:,None,0]-centers[None,:,0])/10
            dxy=np.linalg.norm(expected[:,None,1:]-centers[None,:,1:],axis=2)
            cost=dxy+.25*dz; cost[(dxy>2)|(dz>1)]=1e6
            a,b=linear_sum_assignment(cost)
            assignments={int(i):int(j) for i,j in zip(a,b) if cost[i,j]<1e6}
        for i,g in enumerate(beads):
            j=assignments.get(i)
            cy,cx=expected[i,1:]; radius=int(np.ceil(g['radius_um']/PITCH))+2
            ys=slice(max(0,int(cy)-radius),min(260,int(cy)+radius+1))
            xs=slice(max(0,int(cx)-radius),min(260,int(cx)+radius+1))
            pp=pred[:,ys,xs].sum((1,2)); tp=truth[:,ys,xs].sum((1,2))
            rows.append({**metadata,'bead_index':g['bead_index'],'threshold_fraction':threshold,
                         'detected':j is not None,'centroid_matching':True,
                         'z_error_layers':abs(expected[i,0]-centers[j,0])/10 if j is not None else np.nan,
                         'xy_error_pixels':float(np.linalg.norm(expected[i,1:]-centers[j,1:])) if j is not None else np.nan,
                         'global_false_peak_count':n-len(assignments),'component_count':n,
                         **distribution(pp,tp)})
    return rows,profiles

def t02(pred, truth, metadata):
    geometry=json.loads((exp.DATA/'T02/geometry.json').read_text())['geometry']
    tubes=[g for g in geometry if g['kind']=='tube']
    rows,profiles=[],[]
    for index,g in enumerate(tubes,1):
        path=np.array(g['control_xyz_um'],float)
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))]
        locations=np.arange(0,length[-1]+.25,.5)
        coords=np.array([np.interp(locations,length,path[:,k]) for k in (1,0)])/PITCH
        depth=np.array([map_coordinates(layer,coords,order=1) for layer in pred])
        truth_depth=np.array([map_coordinates(layer,coords,order=1) for layer in truth])
        profile=depth.sum(1); target=truth_depth.sum(1)
        for zi,z in enumerate(Z):
            profiles.append({**metadata,'diagnostic':'T02','region':index,'coordinate':z,
                             'coordinate_unit':'um','value':float(profile[zi]),'gt_value':float(target[zi])})
        for threshold in THRESHOLDS:
            # Localize each centerline sample within two XY pixels and one Z layer.
            support=np.zeros(len(locations),bool)
            zref=float(np.median(path[:,2]))
            for dy in range(-2,3):
                for dx in range(-2,3):
                    if dx*dx+dy*dy>4: continue
                    for zi in np.flatnonzero(abs(Z-zref)<=10):
                        values=map_coordinates(pred[zi],coords+np.array([[dy],[dx]]),order=1)
                        support |= values>=threshold*float(pred.max())
            rows.append({**metadata,'tube':index,'threshold_fraction':threshold,'amplitude':g['amplitude'],
                         'centerline_localized_fraction':float(support.mean()),'false_break':bool((~support).any()),
                         **distribution(profile,target)})
    # The first twelve tube entries are six interrupted strands; quantify gaps
    # from the paired saved endpoints without choosing an ROI from predictions.
    for pair in range(6):
        a=np.asarray(tubes[2*pair]['control_xyz_um'][-1]);b=np.asarray(tubes[2*pair+1]['control_xyz_um'][0])
        fractions=np.linspace(.2,.8,9);xy=a[:2,None]+(b-a)[:2,None]*fractions
        coords=xy[::-1]/PITCH
        zi=int(np.argmin(abs(Z-a[2])))
        gap=map_coordinates(pred[zi],coords,order=1)
        for threshold in THRESHOLDS:
            rows.append({**metadata,'tube':f'gap_{pair+1}','threshold_fraction':threshold,
                         'bridged':bool((gap>=threshold*float(pred.max())).all()),
                         'gap_mean_value':float(gap.mean())})
    return rows,profiles

def run():
    import torch
    torch.set_num_threads(4)
    output=exp.OUTPUT/'analysis'
    output.mkdir(exist_ok=True)
    tables={k:[] for k in ('t03_lines','t04_axial','v03_beads','t02_tubes','fixed_profiles','rl3_metrics')}
    functions={'T02':('t02_tubes',t02),'T03':('t03_lines',t03),'T04':('t04_axial',t04),'V03':('v03_beads',v03)}
    for case in all_cases():
        if case['split'] not in ('validation','test'): continue
        key=DatasetItemKey(case['sample'],case['subset'],case['split'],case['path'])
        truth=_read_targets(key,include_ground_truth=True)['ground_truth'][0]
        raw=load_inference_input(case['path'],case['subset'])
        controls={'mean_rl3':raw['g_mean'][0], 'taylor_rl3':raw['f_var'][0]}
        for method,pred in controls.items():
            tables['rl3_metrics'].append(dict(method=method,case_id=case['id'],sample_id=case['sample'],
                                             subset=case['subset'],split=case['split'],**_structure_row(pred,truth)))
        if case['sample'] not in functions: continue
        name,func=functions[case['sample']]
        for role in ('best','final'):
            for arm in exp.ARMS:
                pred=np.load(exp.OUTPUT/'evaluation'/arm/role/case['id']/'reconstruction.npy')
                metadata=dict(method=f'{arm}_{role}',case_id=case['id'],subset=case['subset'],sample_id=case['sample'],split=case['split'])
                rows,profiles=func(pred,truth,metadata)
                tables[name].extend(rows);tables['fixed_profiles'].extend(profiles)
        for method,pred in controls.items():
            metadata=dict(method=method,case_id=case['id'],subset=case['subset'],sample_id=case['sample'],split=case['split'])
            rows,profiles=func(pred,truth,metadata)
            tables[name].extend(rows);tables['fixed_profiles'].extend(profiles)
    for name,rows in tables.items(): csv_write(output/f'{name}.csv',rows)
    return tables

if __name__=='__main__': run()
