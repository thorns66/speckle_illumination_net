import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base))
os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[2] if len(sys.argv)>2 else '3'
import torch
from adapters import data
from adapters.run import load_trained,validate
from adapters.physics import operator

method=sys.argv[1]
run='smoke_serenet_v2' if method=='serenet' else 'smoke_vcdnet'
args=SimpleNamespace(cpu=False,method=method,checkpoint=str(base/'outputs'/run/'checkpoint_last.pt'))
torch.set_num_threads(4)
model,device,saved=load_trained(args)
op=operator(device) if method=='serenet' else None
value=validate(model,op,device,limit=0)
data.write_json(base/'outputs'/f'full_validation_{method}.json',{'samples':30,'objects':data.SPLITS['validation'],'loss':value,'device':str(device),'gt_used':method=='vcdnet'})
print(method,value,flush=True)
