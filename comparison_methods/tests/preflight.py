import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters import data
from adapters.geometry import split_views,merge_views
from adapters.models import Baseline
from adapters.physics import operator

p=argparse.ArgumentParser(); p.add_argument('--cpu',action='store_true'); p.add_argument('--method',default='serenet'); args=p.parse_args()
torch.set_num_threads(4)
device=torch.device('cpu' if args.cpu else 'cuda:0')
checks={}
for shape in [(260,260),(1029,1421)]:
    image=torch.arange(shape[0]*shape[1],dtype=torch.float32).reshape(1,1,*shape)
    views,mask,s=split_views(image)
    assert torch.equal(merge_views(views,s),image)
    assert int(mask.sum())==shape[0]*shape[1]
    checks[str(shape)]={'views':list(views.shape),'roundtrip_exact':True,'valid_pixels':int(mask.sum())}
with patch.object(data,'truth',side_effect=AssertionError('GT accessed')):
    sample=data.sample('P01',1,supervised=False)
checks['serenet_gt_guard']=True
model=Baseline(args.method,data.contract()).to(device)
image=torch.from_numpy(sample['mean'])[None,None].to(device)
model.eval()
with torch.no_grad(): result=model(image)
assert result.shape==(1,1,10,260,260) and torch.isfinite(result).all()
checks['model']={'shape':list(result.shape),'parameters':sum(p.numel() for p in model.parameters())}
if args.method=='serenet':
    op=operator(device); h=np.load(data.PSF,mmap_mode='r'); errors=[]
    for z,row,col in [(0,122,122),(5,100,101),(9,20,240)]:
        point=torch.zeros(1,1,10,260,260,device=device); point[0,0,z,row,col]=1
        projected=op(point)[0,0].cpu().numpy()
        expected=np.zeros((260,260),np.float32)
        kernel=np.asarray(h[z,row%49,col%49])
        r0=max(0,row-98); r1=min(260,row+99); c0=max(0,col-98); c1=min(260,col+99)
        expected[r0:r1,c0:c1]=kernel[r0-row+98:r1-row+98,c0-col+98:c1-col+98]
        error=float(np.linalg.norm(projected-expected)/max(np.linalg.norm(expected),1e-30))
        assert error<2e-5,(z,row,col,error)
        errors.append(error)
    x=torch.randn(1,1,10,260,260,device=device,requires_grad=True); y=torch.randn(1,1,260,260,device=device)
    lhs=(op(x)*y).sum(); gradient=torch.autograd.grad(lhs,x)[0]; rhs=(x.detach()*gradient).sum()
    error=float(abs(lhs-rhs)/(abs(lhs)+abs(rhs)+1e-5)); assert error<2e-4,error
    checks['point_source_relative_errors']=errors; checks['adjoint_inner_product_error']=error
data.write_json(data.BASE/'outputs'/f'preflight_{args.method}_{device.type}.json',{'passed':True,'device':str(device),'checks':checks})
print(json.dumps(checks),flush=True)
