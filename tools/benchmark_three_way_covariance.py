"""Select an exactly equivalent covariance batch size; never reduce samples."""
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_three_way_validation import inventory
gpu=inventory()[2]
if gpu['busy'] or gpu['memory']>1024:raise RuntimeError('GPU 2 is occupied')
os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch
import yaml
from tools.three_way_experiment import OUTPUT,DATA,load_operator,write_json
from tools.three_way_checks import input_item,relative
from tools.three_way_physics import CorrelatedVariance
from training.multivolume_trainer import _analytic_beta0

torch.set_num_threads(4);torch.cuda.set_device(0)
config=yaml.safe_load((OUTPUT/'e2.yaml').read_text())
operator=load_operator(config,torch.device('cuda:0'))
vm=CorrelatedVariance(operator,config)
item=input_item(DATA/'P08');g0=item['f_var']*_analytic_beta0(operator,item['f_var'],item['input_mean'])[:,None,None,None,None]
vm.active_bank();rows=[];reference=None
for chunk in [8,16,32,64,128,256]:
    vm.chunk=chunk;torch.cuda.reset_peak_memory_stats();g=g0.detach().clone().requires_grad_(True)
    torch.cuda.synchronize();start=time.monotonic()
    try:
        value=vm(g,item['input_mean'])
        weight=torch.linspace(.1,1,value.numel(),device=g.device).reshape(value.shape)
        grad=torch.autograd.grad((torch.log(value+1e-6)*weight).mean(),g)[0]
        torch.cuda.synchronize();seconds=time.monotonic()-start
        if reference is None:reference=(value.detach(),grad.detach())
        row={'chunk':chunk,'seconds':seconds,'peak_gib':torch.cuda.max_memory_allocated()/1024**3,
             'value_relative_l2':relative(value,reference[0]),'gradient_relative_l2':relative(grad,reference[1])}
        rows.append(row);print(row,flush=True)
        del value,grad,g
    except torch.OutOfMemoryError:
        rows.append({'chunk':chunk,'OOM':True});torch.cuda.empty_cache();break
accepted=[r for r in rows if not r.get('OOM') and r['peak_gib']<36 and
          r['value_relative_l2']<1e-4 and r['gradient_relative_l2']<1e-4]
selected=min(accepted,key=lambda r:r['seconds'])
write_json(OUTPUT/'covariance_batch_benchmark.json',{'complete':True,'rows':rows,'selected':selected,
           'sample_count_unchanged':1024,'disable_redundant_outer_checkpoint':True})
print('SELECTED',selected,flush=True)
