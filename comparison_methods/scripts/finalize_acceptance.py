from pathlib import Path
import json
import hashlib
import subprocess
import sys
import torch
import numpy as np
torch.set_num_threads(4)

base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base))
from adapters import data

def read(p): return json.loads((base/p).read_text())
report={'status':'gpu_training_ready','formal_training_started':False,'gpu':'NVIDIA A40; physical index 3; CUDA_VISIBLE_DEVICES maps to cuda:0',
        'methods':{},'scope':'short implementation acceptance, not scientific benchmark results'}
assert (base/'manifests/existing_environment_before.txt').read_bytes()==(base/'manifests/existing_environment_after.txt').read_bytes()
report['original_environment_unchanged']=True
for method in ('serenet','vcdnet'):
    prefix=base/'.envs'/method
    subprocess.run([str(prefix/'bin/python'),'-m','pip','check'],check=True)
    smoke='smoke_serenet_v2' if method=='serenet' else 'smoke_vcdnet'
    first=read(f'outputs/{smoke}/initial_20_steps.json'); resumed=read(f'outputs/{smoke}/run_summary.json')
    fixed=read(f'outputs/overfit_{method}/run_summary.json')
    assert first['steps_this_run']==20 and first['parameters_changed']
    assert resumed['resumed'] and resumed['total_steps']>=22 and resumed['parameters_changed']
    assert fixed['steps_this_run']==100 and fixed['last_loss']<fixed['first_loss']
    restore=read(f'outputs/{smoke}/resume_restore.json')
    assert all(v for k,v in restore.items() if k.endswith('_exact') and v is not None)
    geometry=read(f'outputs/preflight_{method}_cuda.json'); assert geometry['passed']
    full_val=read(f'outputs/full_validation_{method}.json'); assert full_val['samples']==30
    simulation=read(f'outputs/inference_{method}/simulation/inference_summary.json')[0]
    real=read(f'outputs/inference_{method}/real45/inference_summary.json')[0]
    assert simulation['shape']==[10,260,260] and real['shape']==[10,1029,1421]
    metrics=read(f'outputs/inference_{method}/simulation/metrics.json')
    assert len(metrics['rows'])==1 and not metrics['test_complete']
    report['methods'][method]={'initial_training':first,'resume':resumed,'checkpoint_restore':restore,'bitwise_repeated_training_guaranteed':False,
         'fixed_sample':fixed,'preflight':geometry,'full_validation':full_val,'simulation_inference':simulation,'real_inference':real,
         'short_checkpoints_are_not_benchmark_models':True}
report['data_contract_sha256']=data.digest(base/'manifests/data.json')
report['sources']=read('manifests/sources.json')
for r in read('manifests/data.json')['records']:
    assert data.digest(base/'cache/means'/f'{r["object"]}_{r["subset"]:02d}.npy')==r['mean_sha256']
report['all_170_cache_hashes_verified']=True
report['code_sha256']={str(p.relative_to(base)):data.digest(p) for folder in ('adapters','scripts','configs','tests') for p in (base/folder).glob('*') if p.is_file()}
report['readonly_project_dependencies']={str(p):data.digest(p) for p in [base.parent/'tools/mixed_resolution_lfm.py',base.parent/'physics/lfm_operator.py']}
data.write_json(base/'outputs/acceptance.json',report)
lines=['# GPU训练就绪：验收记录','','正式800轮训练尚未启动。以下全部为短链路验收，不能用于论文性能结论。','',
       '| 项目 | SeReNet | VCD-Net |','|---|---|---|',
       '| GPU短训练 | 20步通过 | 20步通过 |','| 续训 | 追加2步通过 | 追加2步通过 |',
       '| 完整验证集 | 30个子集通过 | 30个子集通过 |','| 仿真推理 | 10×260×260 | 10×260×260 |',
       '| 真实45全视野推理 | 10×1029×1421 | 10×1029×1421 |']
for title,key in [('固定样本首步损失','first_loss'),('固定样本第100步损失','last_loss')]:
    lines.append('| '+title+' | '+' | '.join(f'{report["methods"][m]["fixed_sample"][key]:.6g}' for m in ('serenet','vcdnet'))+' |')
lines.extend(['','两个网络均完成20步训练和追加续训；实际开始更新前，对模型权重、优化器、Python/NumPy/PyTorch/CUDA随机状态精确恢复的检查全部通过。每个网络实际累计更新数见JSON。CUDA插值反向存在非确定性，独立重复训练不承诺逐位相同。',
              '原speckle_net环境的前后pip清单完全一致；两个新环境依赖检查通过。',
              '仿真与真实尺寸拆分还原逐值一致；3个点源投影与原PSF一致，伴随内积检查通过。',
              '只对1个仿真测试子集执行了指标导出验收；完整测试需在正式训练后运行。',
              '最初未进行PSF尺度标定的试运行保留在outputs/smoke_serenet中；它不属于验收通过版本，不能续训用于正式实验。',
              '', '详细证据：outputs/acceptance.json。启动命令和适配边界：README_ZH.md。'])
(base/'ACCEPTANCE_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':report['status'],'checkpoint_restore':{m:r['checkpoint_restore'] for m,r in report['methods'].items()}}),flush=True)
