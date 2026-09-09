"""Repair T03's geometry 'group' collision with the challenge group label.

Re-evaluate saved volumes with identical fixed metrics; never re-run inference.
Keep a backup and verify metric values and row order are unchanged.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import h5py
import numpy as np
from tools.report_correlation_challenge import volumes
from tools.v3_compare_local_audit import t03
from tools.v3_compare_evaluation import csv_write
from tools.finalize_correlation_challenge import read
from tools.run_correlation_challenge import save
from datasets.correlation_challenge import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    marker=out/'local_metadata_repair.json'
    if marker.exists():
        assert json.loads(marker.read_text())['after_sha256']==sha256(out/'local_metrics.csv')
        return
    run=json.loads((out/'run.json').read_text())
    manifest=json.loads(Path(run['manifest']).read_text())
    sample=next(s for s in manifest['samples'] if s['sample_id']=='T03')
    with h5py.File(Path(sample['source_dir'])/'prepared.mat') as h:
        truth=np.asarray(h['ground_truth'],dtype=np.float32).transpose(0,2,1)
    corrected=[]
    for role in ('final','best'):
        for group in ['low','high',*[f'random_{i:02d}' for i in range(1,11)]]:
            for method,pred in volumes(out,sample,group,role).items():
                meta={'sample_id':'T03','group':group,'method':method,'checkpoint_role':role}
                rows,_=t03(pred,truth,meta)
                for r in rows:
                    r['geometry_group']=r.pop('group')
                    r['group']=group
                corrected.extend(rows)
    path=out/'local_metrics.csv'
    previous=read(path)
    originals=[r for r in previous if r['sample_id']=='T03']
    assert len(originals)==len(corrected)
    # The fix changes identifiers only, not any numeric/boolean metric.
    for before,after in zip(originals,corrected):
        for key,value in after.items():
            if key in ('group','geometry_group'):continue
            assert before[key]==str(value),(key,before[key],value)
        assert before['group']==str(after['geometry_group'])
    backup=out/'local_metrics_before_metadata_repair.csv'
    assert not backup.exists()
    shutil.copy2(path,backup)
    csv_write(path,corrected+[r for r in previous if r['sample_id']!='T03'])
    save(marker,{'complete':True,'reason':'T03 geometry group overwrote low/high/random group',
                'corrected_rows':len(corrected),'all_metric_values_unchanged':True,
                'before_sha256':sha256(backup),'after_sha256':sha256(path)})


if __name__=='__main__':main()
