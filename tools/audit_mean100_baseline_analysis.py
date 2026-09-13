"""Final CPU provenance/count/QA audit for the independent baseline analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from tools import mean100_baseline_analysis as core


def audit(root):
    manifest,cases=core.setup(root)
    assert json.loads((root/'complete.json').read_text())['complete']
    cache={}
    def digest(path):
        path=Path(path).resolve()
        if path not in cache:cache[path]=core.SHA(path)
        return cache[path]
    inventory=[]
    for case in cases:
        folder=case['path']
        assert digest(folder/'prepared.mat')==manifest['input_fingerprints'][case['id']]['prepared_sha256']
        assert digest(folder/'subsets'/f"subset_{case['subset']:02d}.mat")==manifest['input_fingerprints'][case['id']]['subset_sha256']
        for method in core.METHODS:
            out=root/'evaluation'/method/case['id'];rec=json.loads((out/'complete.json').read_text())
            assert rec['complete'] and rec['inference_target_or_gt_used'] is False
            for file in out.glob('*.npy'):
                array=np.load(file,mmap_mode='r');assert array.shape==(10,260,260) and array.dtype==np.float32 and np.isfinite(array).all()
                if file.name!='effective_correction.npy':assert array.min()>=0
                if file.name=='reconstruction.npy':assert digest(file)==rec['prediction_sha256']
                inventory.append(dict(path=str(file.relative_to(root)),sha256=digest(file)))
            assert (out/'reconstruction.tif').exists()
    for rec in manifest['checkpoints'].values():assert digest(rec['path'])==rec['sha256']
    core.write_csv(root/'analysis/prediction_inventory.csv',inventory)
    source=[]
    # Freeze all project-local Python dependencies loaded by the evaluator/audit.
    for mod in list(sys.modules.values()):
        path=getattr(mod,'__file__',None)
        if not path:continue
        path=Path(path).resolve()
        if path.suffix!='.py' or not path.is_relative_to(core.ROOT):continue
        rel=path.relative_to(core.ROOT)
        if rel.parts[0] not in ('tools','datasets','models','training','physics','losses','utils'):continue
        out=root/'analysis_source_snapshot'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
        source.append(dict(path=str(rel),sha256=digest(path)))
    for rel in ('tools/report_mean100_baseline_analysis.py','tests/test_mean100_baseline_analysis.py'):
        path=core.ROOT/rel;out=root/'analysis_source_snapshot'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
        source.append(dict(path=rel,sha256=digest(path)))
    core.write_csv(root/'analysis/analysis_source_sha256.csv',source)
    spec_path=core.exp.SPARSE_CACHE/'complete.json';spec=json.loads(spec_path.read_text())
    assert spec['threshold']==0 and spec['H_shape'][:3]==[10,49,49] and spec['height']==spec['width']==260
    spec['selected_psf_sha256']=digest(core.ROOT/spec['H_path'])
    spec['sparse_array_sha256']={p.name:digest(p) for p in core.exp.SPARSE_CACHE.glob('*.npy')}
    spec['full_psf_sha256']=digest(manifest['registry']['psf']['path'])
    assert spec['full_psf_sha256']==manifest['registry']['psf']['sha256']
    core.write_json(root/'analysis/physics_fingerprints.json',spec)
    a=root/'analysis';collisions=list(a.rglob('*.collision.json'));alignment=list(a.rglob('*.alignment.json'))
    for path in collisions:
        q=json.loads(path.read_text());assert q['verdict']=='PASS' and q['summary']['fail']==q['summary']['warn']==0
    for path in alignment:assert json.loads(path.read_text())['verdict']=='PASS'
    pages=len(list(a.rglob('*.png')))
    assert len(collisions)==len(alignment)==pages==167,(len(collisions),len(alignment),pages)
    atlas=list((a/'actual_reconstruction_comparison').rglob('atlas.pdf'));assert len(atlas)==28
    expected_pages=sum(json.loads(p.read_text())['pages'] for p in (a/'actual_reconstruction_comparison').rglob('complete.json'))
    assert expected_pages==148
    # PDF glyph checks already ran after every export; every exported page must have one.
    assert len(list(a.rglob('*.audit_pdf_text.py.log')))==pages
    validation=subprocess.run([sys.executable,'/workspace/xyx/.codex/skills/nature-figure/scripts/validate_figure.py',
                               str(core.ROOT/'tools/report_mean100_baseline_analysis.py'),'--json'],capture_output=True,text=True)
    (a/'source_figure_preflight.json').write_text(validation.stdout)
    assert validation.returncode==0
    record=dict(complete=True,source_files=len(source),network_cases=282,rl_cases=188,real_reused_cases=80,
                float_array_files=len(inventory),figure_pages=pages,atlases=28,atlas_pages=148,
                alignment_passes=len(alignment),collision_passes=len(collisions),collision_failures=0,
                data_hashes_reverified=True,checkpoint_hashes_reverified=True,psf_hash_reverified=True,
                full_field_real_reinference=False,no_training_started=True)
    core.write_json(root/'final_acceptance.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();audit(args.output)
