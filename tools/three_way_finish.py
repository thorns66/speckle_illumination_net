"""Complete the comparison with granular summaries, figures and artifact checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import json
import math
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from tools.three_way_experiment import ROOT,OUTPUT,BASELINE,sha256,write_json,PRIORITY
from tools.three_way_report import csv_read,csv_write,report
from tools.priority_validation_common import load_config
import tools.priority_validation_analysis as pa

def mean(rows,key):
    values=[float(r[key]) for r in rows if r.get(key,'') not in ['',None]]
    return float(np.mean(values)) if values else math.nan

def rate(rows,key):
    return float(np.mean([str(r[key]).lower() in ['true','1'] for r in rows])) if rows else math.nan

def grouped(rows,keys):
    groups=collections.defaultdict(list)
    for row in rows:groups[tuple(row.get(k,'') for k in keys)].append(row)
    return groups

def summaries(tables):
    analysis=OUTPUT/'analysis'
    rows=[]
    for keys,values in grouped([r for r in tables['metrics'] if r['split'] in ['test','validation']],['method','sample_id','split']).items():
        rows.append(dict(zip(['method','sample_id','split'],keys),**{k:mean(values,k) for k in
          ['gt_scale_aligned_nrmse','gt_axial_w1_um','local_axial_w1_um','common_mean_loss','common_variance_loss','background_xy_mass_fraction']}))
    csv_write(analysis/'per_object_summary.csv',rows)
    rows=[]
    for keys,values in grouped(tables['point_targets'],['method','threshold','scene_id','repeat']).items():
        singles=[r for r in values if r['kind']=='single']
        rows.append(dict(zip(['method','threshold','scene_id','repeat'],keys),
           single_localization_rate=rate(singles,'matched'),all_points_localization_rate=rate(values,'matched'),
           local_depth_w1_um=mean(values,'local_depth_w1_um'),tail_fraction=mean(values,'tail_mass_outside_pm1'),
           false_peaks_in_test_cells=int(values[0]['unmatched_peaks_in_test_cells'])))
    csv_write(analysis/'per_depth_repeat_summary.csv',rows)
    pairs=[]
    for keys,values in grouped(tables['point_pairs'],['method','threshold','separation_um','axis','ratio']).items():
        pairs.append(dict(zip(['method','threshold','separation_um','axis','ratio'],keys),
            both_localized_rate=rate(values,'both_localized'),separation_rate=rate(values,'separated'),n=len(values)))
    csv_write(analysis/'pair_resolution_summary.csv',pairs)
    lines=[]
    for keys,values in grouped(tables['lines'],['method','threshold','gap_um','angle_deg','amplitude']).items():
        lines.append(dict(zip(['method','threshold','gap_um','angle_deg','amplitude'],keys),
            false_gap_rate=rate(values,'false_gap'),bridge_rate=rate(values,'bridged'),
            endpoint_retention=mean(values,'endpoint_retention_fraction'),coverage=mean(values,'coverage_fraction'),n=len(values)))
    csv_write(analysis/'line_quality_summary.csv',lines)

def line_profiles(methods):
    _,config=load_config();pitch=config['acquisition']['object_pixel_pitch_um'];distance=np.linspace(-16,16,129)
    rows=[];definitions=pa.line_definitions(config)
    for depth in [20,40,60,80,90]:
        for repeat in [1,2,3]:
            case=f'lines_z{depth:03d}_r{repeat:02d}';zi=depth//10-1
            for method in methods:
                experiment,role=method.split('_');volume=np.load(OUTPUT/'evaluation'/experiment/role/case/'reconstruction.npy')
                for definition in definitions:
                    theta=np.deg2rad(float(definition['angle_deg']))
                    x=(float(definition['x_um'])+np.cos(theta)*distance)/pitch
                    y=(float(definition['y_um'])+np.sin(theta)*distance)/pitch
                    profile=np.max([map_coordinates(volume,np.vstack((np.full_like(x,z),y,x)),order=1,mode='constant',cval=0)
                                    for z in range(max(0,zi-1),min(10,zi+2))],axis=0)
                    rows.extend({'method':method,'case_id':case,**definition,'distance_um':float(d),
                                 'value':float(v),'whole_volume_max':float(volume.max()),'z_search':'truth layer plus/minus one'}
                                for d,v in zip(distance,profile))
    csv_write(OUTPUT/'analysis/line_fixed_profiles.csv',rows)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    selected=[d for d in definitions if (d['gap_um'],d['angle_deg'],d['amplitude']) in
               [(0,0,1),(0,45,.5),(4,0,1),(8,0,1),(12,0,1),(8,90,.5)]]
    for ax,definition in zip(axes.ravel(),selected):
        for method in methods:
            part=[r for r in rows if r['case_id']=='lines_z060_r01' and r['method']==method and r['cell_id']==definition['cell_id']]
            ax.plot([r['distance_um'] for r in part],[r['value']/max(r['whole_volume_max'],1e-30) for r in part],label=method)
        gap=definition['gap_um']
        if gap:ax.axvspan(-gap/2,gap/2,color='gray',alpha=.2)
        ax.set_title(f"gap={gap} um; angle={definition['angle_deg']}; amplitude={definition['amplitude']}")
        ax.set_xlabel('Fixed distance along line, um');ax.set_ylabel('Value / one whole-volume maximum')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUTPUT/'analysis/line_fixed_profiles.png',dpi=160);plt.close(fig)

def brightness_figure(rows,methods):
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,sample in zip(axes,['points_z060_r01','lines_z060_r01','P09']):
        for method in methods:
            selected=[r for r in rows if r['sample_id']==sample and r['method']==method]
            reference=float(next(r for r in selected if float(r['gain'])==1)['output_sum'])
            selected.sort(key=lambda r:float(r['gain']))
            ax.plot([float(r['gain']) for r in selected],[float(r['output_sum'])/reference for r in selected],marker='o',label=method)
        ax.plot([0,2],[0,2],'k--',label='ideal');ax.set_title(sample);ax.set_xlabel('Raw input gain');ax.set_ylabel('Output sum / output sum at gain 1')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUTPUT/'analysis/brightness_response.png',dpi=160);plt.close(fig)

def verify_artifacts():
    preflight=json.loads((OUTPUT/'preflight.json').read_text());changed=[p for p,h in preflight['source_sha256'].items() if sha256(ROOT/p)!=h]
    if changed:raise ValueError(f'Frozen sources changed: {changed}')
    assert sha256(BASELINE/'checkpoint_best.pt')==preflight['baseline_best_sha256']
    assert sha256(BASELINE/'checkpoint_last.pt')==preflight['baseline_final_sha256']
    records=[]
    for experiment in ['baseline','e1','e2','e3']:
        folder=OUTPUT/'evaluation'/experiment
        for role in ['best','final']:
            weight_folder=BASELINE if experiment=='baseline' else OUTPUT/experiment
            weight_hash=sha256(weight_folder/('checkpoint_best.pt' if role=='best' else 'checkpoint_last.pt'))
            markers=list((folder/role).glob('*/complete.json'))
            assert len(markers)==114,(experiment,role,len(markers))
            for marker in markers:
                record=json.loads(marker.read_text());path=marker.parent/'reconstruction.npy';value=np.load(path)
                assert value.shape==(10,260,260) and np.isfinite(value).all() and (value>=0).all()
                assert sha256(path)==record['prediction_sha256']
                assert record['checkpoint_sha256']==weight_hash
            records.append({'method':experiment+'_'+role,'predictions':len(markers),
               'brightness_predictions':len(list((folder/role/'brightness').glob('*/*.npy')))})
            assert records[-1]['brightness_predictions']==15
    assert json.loads((OUTPUT/'brightness_inputs/verification.json').read_text())['passed']
    assert json.loads((OUTPUT/'mismatch_evaluation/complete.json').read_text())['prediction_count']==24
    write_json(OUTPUT/'analysis/final_artifact_verification.json',{'passed':True,'frozen_sources_unchanged':True,
          'original_baseline_weights_unchanged':True,'main_predictions':912,'brightness_predictions':120,
          'NA_mismatch_predictions':24,'models':records,'files_hashed':True})

def run():
    report()
    names=['metrics','point_targets','point_pairs','lines','brightness','covariance_audit']
    tables={name:csv_read(OUTPUT/'analysis'/(name+'.csv')) for name in names}
    methods=[r['method'] for r in csv_read(OUTPUT/'analysis/summary.csv')]
    summaries(tables);line_profiles(methods);brightness_figure(tables['brightness'],methods);verify_artifacts()
    from tools.three_way_conclusions import write_report
    write_report()

if __name__=='__main__':run()
