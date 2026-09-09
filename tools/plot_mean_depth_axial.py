"""Render saved four-arm reconstructions in axial views; never run inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from datasets.matlab_multivolume_dataset import DatasetItemKey, _read_targets, load_inference_input
from tools import mean_depth_experiment as exp
from tools.mean_depth_evaluation import all_cases
from tools.mean_depth_report import LABELS, write_csv
from tools.v3_compare_evaluation import _um_slice
from tools.priority_validation_common import load_config, sha256
from tools.priority_validation_analysis import axial_targets

OUT = exp.OUTPUT / 'analysis/axial_comparison'
Z = np.arange(10., 101., 10.)
PITCH = 220 / 4 / 49
METHODS = ['GT', 'mean_rl3', 'taylor_rl3', 'source_b', *exp.ARMS]
NAMES = {'GT': 'GT', 'mean_rl3': 'Mean-RL3', 'taylor_rl3': 'Taylor-RL3', **LABELS}
COLORS = ['black', '#999999', '#ab7a23', '#8844aa', '#1676bb', '#ed7d21', '#299944', '#d52e38']


def load_case(case, role, sources):
    key = DatasetItemKey(case['sample'], case['subset'], case['split'], case['path'])
    truth = _read_targets(key, include_ground_truth=True)['ground_truth'][0]
    raw = load_inference_input(case['path'], case['subset'])
    arrays = {'GT': truth, 'mean_rl3': raw['g_mean'][0], 'taylor_rl3': raw['f_var'][0]}
    for method in METHODS[3:]:
        folder = exp.SOURCE_OUTPUT / 'evaluation/e3/final' if method == 'source_b' else exp.OUTPUT / 'evaluation' / method / role
        path = folder / case['id'] / 'reconstruction.npy'
        arrays[method] = np.load(path, allow_pickle=False)
        sources[str(path)] = sha256(path)
    for path in (case['path'] / 'prepared.mat', case['path'] / 'subsets' / f"subset_{case['subset']:02d}.mat"):
        if path.exists() and str(path) not in sources:
            sources[str(path)] = sha256(path)
    for name, array in arrays.items():
        if array.shape != (10, 260, 260) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"Invalid volume: {case['id']} {name}")
    return arrays


def save_figure(fig, name, book, manifest, case, role):
    path = OUT / f'{name}.png'
    fig.savefig(path, dpi=160)
    book.savefig(fig)
    plt.close(fig)
    manifest.append({'case_id': case['id'], 'role': role, 'figure': path.name, 'sha256': sha256(path)})


def side_views(case, role, arrays, book, manifest):
    # Integrate each transverse axis. This is an orthogonal projection, not
    # a single slice and not proof that two signals share the same XY location.
    projections = {name: (a.sum(axis=1), a.sum(axis=2)) for name, a in arrays.items()}
    for common in (False, True):
        fig, axes = plt.subplots(2, 8, figsize=(24, 7), layout='constrained')
        network_scale = max(float(p.max()) for name in METHODS[3:] for p in projections[name])
        for column, name in enumerate(METHODS):
            pair = projections[name]
            scale = network_scale if common and column >= 3 else max(float(p.max()) for p in pair)
            for row, values in enumerate(pair):
                ax = axes[row, column]
                ax.imshow(values / max(scale, 1e-30), cmap='magma', vmin=0, vmax=1,
                          interpolation='nearest', origin='lower', aspect='equal',
                          extent=(-PITCH/2, 259.5*PITCH, 5, 105))
                ax.set_xlabel('X (µm)' if row == 0 else 'Y (µm)')
                ax.set_ylabel('Z (µm)' if column == 0 else '')
                ax.set_yticks(Z[::2])
                if row == 0:
                    ax.set_title(NAMES[name], fontsize=10)
        mode = '五个网络预测共同亮度；GT/RL3各自归一化' if common else '各方法独立归一化；同方法两投影共用尺度'
        fig.suptitle(f"{case['id']} · {role} · 上：XZ沿Y求和，下：YZ沿X求和\n{mode}；原生10层，无Z插值", fontsize=13)
        save_figure(fig, f"{case['id']}_{role}_XZ_YZ_{'shared' if common else 'normalized'}", book, manifest, case, role)


def regions(case, priority):
    if case['sample'] == 'T03':
        groups = json.loads((exp.DATA / 'T03/geometry.json').read_text())['geometry']
        result = []
        for i, g in enumerate(groups, 1):
            x0, y0, x1, y1 = np.asarray(g['bbox_xy_one_based'], int) - 1
            result.append((f"{i}: {g['orientation']} {g['width_um']:.2f}µm", slice(y0,y1+1), slice(x0,x1+1), [g['depth_um']]))
        return result
    if case['sample'] == 'T04':
        result = []
        for g in json.loads((exp.DATA / 'T04/geometry.json').read_text())['geometry']:
            ys, xs = _um_slice(g['roi_bounds_xy_um'], PITCH)
            depths = np.atleast_1d(g['z_um']).tolist()
            note = '；10µm不判分离' if g['separation_um'] == 10 else ''
            result.append((f"{g['region_id']}: {depths}µm{note}", ys, xs, depths))
        return result
    result = []
    for cell in range(1, 11):
        targets = [t for t in axial_targets(priority) if t['cell_id'] == cell]
        target = targets[0]
        # Fixed 12 um half-width, chosen from geometry before looking at output.
        x, y = target['x_um'], target['y_um']
        ys, xs = _um_slice([x-12,y-12,x+12,y+12], PITCH)
        depths = [t['z_um'] for t in targets]
        ratio = target.get('ratio', 1)
        result.append((f"cell {cell}: {depths}µm, 比例1:{ratio:g}", ys, xs, depths))
    return result


def profiles(case, role, arrays, priority, book, manifest, rows, roi_rows):
    entries = regions(case, priority)
    nrows = (len(entries) + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(18, 3.0*nrows+1), squeeze=False)
    for ax, (label, ys, xs, expected) in zip(axes.flat, entries):
        roi_rows.append({'case_id':case['id'], 'role':role, 'region':label,
                         'y_start':ys.start,'y_stop_exclusive':ys.stop,
                         'x_start':xs.start,'x_stop_exclusive':xs.stop, 'expected_depths_um':expected})
        for z in expected:
            ax.axvline(z, color='black', alpha=.12, linewidth=2)
        for name, color in zip(METHODS, COLORS):
            raw = arrays[name][:, ys, xs].astype(np.float64).sum(axis=(1, 2))
            mass = float(raw.sum())
            fraction = raw / max(mass, 1e-30)
            style = '--' if name in ('mean_rl3','taylor_rl3') else '-'
            ax.plot(Z, fraction, style, marker='o', markersize=3,
                    linewidth=2 if name == 'GT' else 1.3, color=color, label=NAMES[name])
            assert mass == 0 or np.isclose(fraction.sum(), 1)
            for z, value, part in zip(Z, raw, fraction):
                rows.append({'case_id':case['id'], 'role':role, 'method':name,
                             'region':label,'z_um':z,'raw_layer_sum':value,
                             'roi_total_mass':mass,'layer_mass_fraction':part})
        ax.set_title(label, fontsize=9)
        ax.set_xticks(Z[::2]); ax.set_ylim(-.025, 1.05)
        ax.set_xlabel('Z (µm)'); ax.set_ylabel('局部总量中该层占比')
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(entries):]:
        ax.axis('off')
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.957), ncol=4, fontsize=9)
    fig.suptitle(f"{case['id']} · {role} · 固定ROI原生十层深度分布\n圆点为真实采样；连线仅引导阅读；近零目标须结合CSV原始总量", fontsize=12)
    fig.tight_layout(rect=(0,0,1,.88))
    save_figure(fig, f"{case['id']}_{role}_depth_profiles", book, manifest, case, role)


def main():
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [c for c in all_cases() if c['sample'] in ('T02','T03','T04','V03') or c.get('family') == 'axial_pairs']
    _, priority = load_config()
    sources, manifest, rows, roi_rows = {}, [], [], []
    for role in ('best','final'):
        with PdfPages(OUT / f'轴向对比_{role}.pdf') as book:
            for case in cases:
                show_side = case['subset'] == 1
                show_profiles = case['sample'] in ('T03','T04') or case.get('family') == 'axial_pairs'
                if not show_side and not show_profiles:
                    continue
                arrays = load_case(case, role, sources)
                if show_side:
                    side_views(case, role, arrays, book, manifest)
                if show_profiles:
                    profiles(case, role, arrays, priority, book, manifest, rows, roi_rows)
                print(f"Rendered {case['id']} {role}", flush=True)
    write_csv(OUT / 'fixed_depth_profiles.csv', rows)
    write_csv(OUT / 'fixed_rois.csv', roi_rows)
    write_csv(OUT / 'figure_manifest.csv', manifest)
    for p in (exp.DATA/'T03/geometry.json', exp.DATA/'T04/geometry.json', ROOT/'configs/priority_validation_20260907.yaml', Path(__file__)):
        sources[str(p)] = sha256(p)
    # No training/report/checkpoint files are changed by this supplementary entry.
    for path, digest in sources.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f'Input changed while rendering: {path}')
    record = {'passed':True,'png_count':len(manifest),'profile_rows':len(rows),
              'z_um':Z.tolist(),'pixel_pitch_um':PITCH,'roles':['best','final'],
              'sources':sources,'artifacts':{p.name:sha256(p) for p in OUT.iterdir() if p.suffix in ('.png','.pdf','.csv')}}
    (OUT/'acceptance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    lines = ['# 四组实验轴向对比补充图','',
             '上排XZ沿Y求和，下排YZ沿X求和；投影会叠加不同横向位置，不能独自判断同位置双层是否分开。局部曲线才用于检查固定位置的层间分配。', '',
             '深度为原生10–100 µm、10 µm间隔；不插值、不逐层调亮。normalized：每个方法两幅投影共用一个尺度。shared：B与R0–R3共用一个尺度，GT及RL3各自归一化（单位不同）。', '',
             '局部曲线按每个ROI总量归一化，表示能量分到了哪些层；原始逐层量与总量保存在CSV。弱目标或漏检不能仅凭归一化曲线判断。T03/T04 ROI沿用原几何；轴向点对使用各中心±12 µm固定窗口，仅用于显示。', '',
             'T03/T04十个重叠子集各自出图，不当作独立重复；轴向点对三次独立采集分别出图。10 µm双层不作谷值分辨判断。所有图来自已保存预测，没有重训或更换checkpoint。', '',
             '- [Final完整轴向图册](轴向对比_final.pdf)', '- [Best完整轴向图册](轴向对比_best.pdf)',
             '- [局部曲线原始数据](fixed_depth_profiles.csv)', '- [固定ROI](fixed_rois.csv)', '', '## 逐图索引','']
    lines.extend(f"- [{r['case_id']} · {r['role']} · {r['figure']}]({r['figure']})" for r in manifest)
    (OUT/'README_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'png_count':len(manifest),'profile_rows':len(rows),'output':str(OUT)},ensure_ascii=False))


if __name__ == '__main__':
    main()
