from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import BASE, PSF

PERIOD=49


def split_views(image):
    b,c,h,w=image.shape
    if c!=1: raise ValueError('Expected one mean sensor image')
    hp=((h+48)//49)*49; wp=((w+48)//49)*49
    padded=F.pad(image,(0,wp-w,0,hp-h))
    mask=F.pad(torch.ones_like(image),(0,wp-w,0,hp-h))
    def rearrange(x):
        return x.reshape(b,1,hp//49,49,wp//49,49).permute(0,1,3,5,2,4).reshape(b,2401,hp//49,wp//49)
    return rearrange(padded),rearrange(mask),(h,w)


def merge_views(views,shape):
    b,a,h,w=views.shape
    if a!=2401: raise ValueError('2401 views required')
    return views.reshape(b,49,49,h,w).permute(0,3,1,4,2).reshape(b,1,h*49,w*49)[...,:shape[0],:shape[1]]


def prepare_shifts():
    # A unit emitter at object coordinate (122,122), phase (24,24).
    # H[Z,phase_row,phase_col,dy,dx] is centered at kernel pixel (98,98).
    h=np.load(PSF,mmap_mode='r')
    shifts=np.zeros((2401,10,2),np.float32)
    valid=np.zeros((2401,10),bool)
    # Absolute response coordinates are 122 + (kernel_position - 98).
    for a in range(49):
        rows=np.arange(197)[(np.arange(197)+24)%49==a]
        for b in range(49):
            cols=np.arange(197)[(np.arange(197)+24)%49==b]
            weights=np.asarray(h[:10,24,24])[:,rows][:,:,cols]
            energy=weights.sum((1,2),dtype=np.float64)
            y=((rows+24-a)/49)[None,:,None]; x=((cols+24-b)/49)[None,None,:]
            # All angles refocus to lens center 122 = 2*49+24, i.e. coarse coordinate (2,2).
            cy=(weights*y).sum((1,2))/np.maximum(energy,1e-30)
            cx=(weights*x).sum((1,2))/np.maximum(energy,1e-30)
            shifts[a*49+b,:,0]=np.where(energy>0,cx-2,0)
            shifts[a*49+b,:,1]=np.where(energy>0,cy-2,0)
            valid[a*49+b]=energy>0
    np.save(BASE/'cache/psf_shifts.npy',shifts)
    np.save(BASE/'cache/psf_view_valid.npy',valid)
    return shifts


class Refocus(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift',torch.from_numpy(np.load(BASE/'cache/psf_shifts.npy')),persistent=False)

    def forward(self,views,mask):
        batch,angles,h,w=views.shape
        yy,xx=torch.meshgrid(torch.arange(h,device=views.device),torch.arange(w,device=views.device),indexing='ij')
        grid=torch.stack((xx,yy),-1).float()[None,None]+self.shift[:,:,None,None,:]
        grid[...,0]=(grid[...,0]+.5)*2/w-1
        grid[...,1]=(grid[...,1]+.5)*2/h-1
        # Each input angle is sampled at each physical depth; zero padding is explicit.
        flat=grid.reshape(angles*10,h,w,2)
        outputs=[]
        for i in range(batch):
            source=views[i,:,None].expand(-1,10,-1,-1).reshape(angles*10,1,h,w)
            weight=mask[i,:,None].expand(-1,10,-1,-1).reshape_as(source)
            result=F.grid_sample(source,flat,align_corners=False,padding_mode='zeros')
            denom=F.grid_sample(weight,flat,align_corners=False,padding_mode='zeros')
            result=torch.where(denom>1e-6,result/denom.clamp_min(1e-6),0)
            outputs.append(result.reshape(angles,10,h,w))
        return torch.stack(outputs)
