"""Export explicit RL3 volumes, summarize local resolution and audit delivery."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import h5py
import numpy as np
import tifffile
from datasets.correlation_challenge import load_challenge_input, sha256
from tools.run_correlation_challenge import save
from tools.v3_compare_evaluation import csv_write


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert json.loads((out/'complete.json').read_text())['complete']
    run = json.loads((out/'run.json').read_text())
    manifest = json.loads(Path(run['manifest']).read_text())
    exports=[]
    for sample in manifest['samples']:
        for group in ('low','high'):
            raw=load_challenge_input(run['manifest'],sample['sample_id'],group)
            for method,field in [('mean_rl3','g_mean'),('taylor_rl3','f_var')]:
                dest=out/'rl3_reconstructions'/sample['sample_id']/group/method
                dest.mkdir(parents=True,exist_ok=True)
                volume=raw[field][0]
                np.save(dest/'reconstruction.npy',volume)
                tifffile.imwrite(dest/'reconstruction_float.tif',volume,metadata={'axes':'ZYX','z_um':list(range(10,101,10))})
                exports.append({'sample_id':sample['sample_id'],'group':group,'method':method,
                    'input_indices':raw['input_indices'].tolist(),'path':str(dest/'reconstruction.npy'),
                    'sha256':sha256(dest/'reconstruction.npy'),'shape':list(volume.shape),
                    'intensity':'retained scale; Taylor square root, Mean unchanged'})
    locals_=read(out/'local_metrics.csv')
    local_summary=[]
    for sample in ('T03','T04'):
        for role in ('final','best'):
            for group in ('low','high','random'):
                for method in ('mean_rl3','taylor_rl3','baseline','e3','e3_mean005'):
                    selected=[r for r in locals_ if r['sample_id']==sample and r['checkpoint_role']==role
                        and r['method']==method and float(r['threshold_fraction'])==.1
                        and (r['group'].startswith('random_') if group=='random' else r['group']==group)]
                    if sample=='T04':
                        eligible=[r for r in selected if r['separation_applicable']=='True']
                    else:
                        eligible=selected
                    assert eligible
                    local_summary.append({'sample_id':sample,'checkpoint_role':role,'group':group,'method':method,
                        'eligible_regions':len(eligible),'separated_rate':float(np.mean([r['separated']=='True' for r in eligible])),
                        'local_depth_w1_um':float(np.mean([float(r['local_depth_w1_um']) for r in selected])),
                        'criterion':'T03 three-line groups' if sample=='T03' else 'T04 native 20/30/40 um only; 10 um excluded'})
    csv_write(out/'local_summary.csv',local_summary)
    csv_write(out/'rl3_volume_inventory.csv',exports)
    predictions=list((out/'predictions').glob('*/*/*/*/complete.json'))
    assert len(predictions)==36
    for path in predictions:
        record=json.loads(path.read_text())
        volume_path=path.parent/'reconstruction.npy'
        assert sha256(volume_path)==record['prediction_sha256']
        a=np.load(volume_path)
        assert a.shape==(10,260,260) and np.isfinite(a).all() and a.min()>=0
    metrics=read(out/'metrics.csv')
    assert len(metrics)==360
    assert sum(r['group'] in ('low','high') for r in metrics)==60
    figures=list((out/'figures').rglob('*.png'))
    assert len(figures)>=61
    summary=read(out/'object_macro.csv')
    per_object=read(out/'per_object.csv')
    lines=['# 低／高相关子集：最终结果速览','',
        '已完成 6 个固定子集、12 个 Mean/Taylor-RL3 重建体，以及三组网络各自 final/best 共 36 个三维预测。',
        '主结论以第 400 步 final 为准；所有方法共用帧编号，不重训、不重选 checkpoint。','',
        '| 方法 | 低相关 NRMSE ↓ | 高相关 NRMSE ↓ | 低相关 W1 μm ↓ | 高相关 W1 μm ↓ |',
        '|---|---:|---:|---:|---:|']
    for method in ('mean_rl3','taylor_rl3','baseline','e3','e3_mean005'):
        pair={r['group']:r for r in summary if r['checkpoint_role']=='final' and r['method']==method}
        lo,hi=pair['low'],pair['high']
        lines.append(f"| {method} | {float(lo['gt_scale_aligned_nrmse']):.4f} | {float(hi['gt_scale_aligned_nrmse']):.4f} | {float(lo['gt_axial_w1_um']):.2f} | {float(hi['gt_axial_w1_um']):.2f} |")
    lines += ['', '## 实际分辨能力（final）','',
        '| 方法 | T03 低相关三线分开率 | T03 高相关 | T04 低相关双层分开率 | T04 高相关 |',
        '|---|---:|---:|---:|---:|']
    for method in ('mean_rl3','taylor_rl3','baseline','e3','e3_mean005'):
        rr={(r['sample_id'],r['group']):r['separated_rate'] for r in local_summary
            if r['checkpoint_role']=='final' and r['method']==method}
        lines.append('| '+method+' | '+' | '.join(f"{rr[s,g]:.1%}" for s,g in [('T03','low'),('T03','high'),('T04','low'),('T04','high')])+' |')
    lines += ['', 'T04 分开率只包括原生 20/30/40 μm，10 μm 相邻层不通过插值算作分开。',
        '', '## 低相关是否全面更好？','']
    for method in ('mean_rl3','taylor_rl3','baseline','e3','e3_mean005'):
        wins=[]
        for sample in ('T02','T03','T04'):
            rr={r['group']:r for r in per_object if r['sample_id']==sample and r['method']==method and r['checkpoint_role']=='final'}
            if all(float(rr['low'][k])<float(rr['high'][k]) for k in ('gt_scale_aligned_nrmse','gt_axial_w1_um')):
                wins.append(sample)
        lines.append(f"- {method}：低相关在对齐 NRMSE 与 W1 两项同时改善的对象为 {', '.join(wins) if wins else '无'}（{len(wins)}/3）。")
    lines += ['', '不能把低相关直接等同于有效信息更多：选帧也可能改变亮度分布和结构覆盖；本结果是从 100 帧择 10 帧的固定测试，不是独立重复或显著性证明。',
        '', '## 文件入口','',
        '- [完整报告](REPORT_ZH.md)、[逐子集指标](metrics.csv)、[对象等权平均](object_macro.csv)、[局部分辨指标](local_summary.csv)。',
        '- [12 个 RL3 重建体清单](rl3_volume_inventory.csv)：每个都有保留尺度的 NPY 与浮点 TIFF，不是截图。',
        '- 网络三维体位于 predictions；低／高相关的原生层、共同尺度和完整三维体归一化图位于 figures。']
    for sample in ('T02','T03','T04'):
        lines.append(f'- {sample}：[XY 对比](figures/{sample}/final/shape_xy.png)、[XZ 深度对比](figures/{sample}/final/shape_xz.png)、[全部原生层](figures/{sample}/final/shape_native_layers.png)。')
    (out/'FINAL_SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n')
    save(out/'delivery_acceptance.json',{'complete':True,'challenge_subsets':6,'rl3_volumes':12,
        'network_volumes':36,'challenge_metric_rows':60,'random_reference_rows':300,'figures':len(figures),
        'verified_prediction_hashes':True,'rl3_export_hashes':exports})
    print('DELIVERY COMPLETE',len(figures),'figures')


if __name__=='__main__':
    main()
