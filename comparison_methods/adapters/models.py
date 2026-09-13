from __future__ import annotations
import ast
import types
import torch
from torch import nn
from .data import BASE
from .geometry import Refocus, split_views


def upstream_module(name):
    path=BASE/'upstream/SeReNet/save/month_202505/config_tag/code/models'/f'{name}.py'
    tree=ast.parse(path.read_text())
    # Remove only registry and unused helper imports, avoiding collisions with this project's models/utils.
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='models')
               and not (isinstance(n,ast.Import) and any(a.name=='utils' for a in n.names))]
    for n in tree.body:
        if isinstance(n,ast.ClassDef):
            n.decorator_list=[d for d in n.decorator_list if not (isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='register')]
    module=types.ModuleType('baseline_upstream_'+name)
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    return module


class Baseline(nn.Module):
    def __init__(self,method,normalization):
        super().__init__(); self.method=method
        self.input_scale=float(normalization['input_scale'])
        self.output_scale=float(normalization['serenet_output_scale'])
        self.target_scale=float(normalization['vcd_target_scale']) if method=='vcdnet' else None
        if method=='serenet':
            self.net=upstream_module('serenet').SERENET(inChannels=2401,outChannels=10,reset_param=True)
            self.refocus=Refocus()
        elif method=='vcdnet':
            self.net=upstream_module('vcdnet').VCDNET(inChannels=2401,outChannels=10,channels_interp=128)
        else: raise ValueError(method)

    def forward(self,image):
        views,mask,shape=split_views(image/self.input_scale)
        if self.method=='serenet':
            result=self.net(self.refocus(views,mask),49)
            if isinstance(result,tuple): result=result[0]
            volume=result[...,:shape[0],:shape[1]]*self.output_scale
        else:
            # Match upstream tanh output to a fixed physical [0, train_max * 1.05] range.
            result=self.net(views,49)
            volume=(result[...,:shape[0],:shape[1]]+1)*(.5*self.target_scale)
        return volume
