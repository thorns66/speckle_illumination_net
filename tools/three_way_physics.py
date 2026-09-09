"""Exact sparse LFM projection and streamed correlated variance for E1/E2/E3.

No PSF values are thresholded: only exact zeros are omitted. Each object voxel
is one column, with precisely the cropped MATLAB conv2(...,'same') PSF.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


def build_sparse_cache(h_path: Path, output: Path, height=260, width=260):
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'complete.json'
    if marker.exists():
        return json.loads(marker.read_text())
    h = np.load(h_path, mmap_mode='r')
    nz, period, _, kh, kw = h.shape
    if kh % 2 != 1 or kw % 2 != 1:
        raise ValueError('Exact sparse construction currently requires odd kernels')
    count = np.zeros(nz*height*width, dtype=np.int64)
    kernels = []
    for z in range(nz):
        for a in range(period):
            for b in range(period):
                kernel = h[z,a,b]
                ky,kx = np.nonzero(kernel)
                ky=ky.astype(np.int32)-kh//2; kx=kx.astype(np.int32)-kw//2
                val=np.asarray(kernel[ky+kh//2,kx+kw//2])
                yy,xx=np.meshgrid(np.arange(a,height,period),np.arange(b,width,period),indexing='ij')
                yy=yy.ravel(); xx=xx.ravel(); ids=z*height*width+yy*width+xx
                sy=yy[:,None]+ky; sx=xx[:,None]+kx
                valid=(sy>=0)&(sy<height)&(sx>=0)&(sx<width)
                count[ids]=valid.sum(1)
                kernels.append((ids,yy,xx,ky,kx,val))
    indptr=np.concatenate(([0],np.cumsum(count)))
    if indptr[-1]>=2**31:
        raise ValueError('CSR requires int64 indexing for this geometry')
    indptr=indptr.astype(np.int32)
    indices=np.empty(int(indptr[-1]),np.int32); values=np.empty(int(indptr[-1]),np.float32)
    for ids,yy,xx,ky,kx,val in kernels:
        sy=yy[:,None]+ky; sx=xx[:,None]+kx
        valid=(sy>=0)&(sy<height)&(sx>=0)&(sx<width)
        dest=indptr[ids,None]+np.cumsum(valid,axis=1)-1
        indices[dest[valid]]=(sy*width+sx)[valid]
        values[dest[valid]]=np.broadcast_to(val,valid.shape)[valid]
    transpose=sp.csr_matrix((values,indices,indptr),shape=(nz*height*width,height*width))
    forward=transpose.transpose().tocsr()
    for name,matrix in [('forward',forward),('transpose',transpose)]:
        for field in ['data','indices','indptr']:
            np.save(output/f'{name}_{field}.npy',getattr(matrix,field))
    record={'complete':True,'H_path':str(h_path),'H_shape':list(h.shape),'height':height,
            'width':width,'nnz':int(values.size),'threshold':0,'built_unix':time.time()}
    marker.write_text(json.dumps(record,indent=2))
    return record


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, owner, squared):
        ctx.owner=owner;ctx.squared=squared;ctx.shape=value.shape
        matrix=owner._a2 if squared else owner._a
        return torch.sparse.mm(matrix,value.flatten(1).T.contiguous()).T.reshape(
            value.shape[0],1,owner.height,owner.width)

    @staticmethod
    def backward(ctx, gradient):
        owner=ctx.owner
        matrix=owner._at2 if ctx.squared else owner._at
        value=torch.sparse.mm(matrix,gradient.flatten(1).T.contiguous()).T.reshape(ctx.shape)
        return value,None,None


class SparseLFM(nn.Module):
    def __init__(self, original, cache: Path):
        super().__init__()
        self.register_buffer('H',original.H,persistent=False)
        self.phase_chunk_size=original.phase_chunk_size
        self.mode='exact_sparse'
        spec=json.loads((cache/'complete.json').read_text())
        self.height=spec['height'];self.width=spec['width']
        self.num_depths=int(self.H.shape[0]);self.phase_period=int(self.H.shape[1])
        rows=self.height*self.width;cols=rows*self.num_depths
        for name,prefix,shape in [('_a','forward',(rows,cols)),('_at','transpose',(cols,rows))]:
            arrays={field:torch.as_tensor(np.load(cache/f'{prefix}_{field}.npy',mmap_mode='c'),
                     device=self.H.device) for field in ['indptr','indices','data']}
            matrix=torch.sparse_csr_tensor(arrays['indptr'],arrays['indices'],arrays['data'],size=shape)
            self.register_buffer(name,matrix,persistent=False)
            square=torch.sparse_csr_tensor(arrays['indptr'],arrays['indices'],arrays['data'].square(),size=shape)
            self.register_buffer(name+'2',square,persistent=False)

    def forward(self,volume):
        return _Project.apply(volume,self,False)

    def forward_squared(self,volume):
        return _Project.apply(volume,self,True)

    def adjoint(self,sensor):
        return torch.sparse.mm(self._at,sensor.flatten(1).T.contiguous()).T.reshape(
            sensor.shape[0],1,self.num_depths,self.height,self.width)


def generate_bank(path: Path, seeds, *, na=.05, frames_per_seed=64, size=260,
                  wavelength_um=.488, pixel_um=1.125, z_um=tuple(range(10,101,10))):
    """Independent random phases; one common complex field propagated through Z.

    CPU NumPy generates deterministic phases. No object or measured frame is read.
    Floating raw intensities retain one fixed illumination unit (no frame maxima).
    """
    shape=(len(seeds)*frames_per_seed,len(z_um),size,size)
    bank=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
    padded=2*size
    freq=np.fft.fftshift(np.fft.fftfreq(padded,d=pixel_um))
    fy,fx=np.meshgrid(freq,freq,indexing='ij')
    pupil=(fx*fx+fy*fy)<=(na/wavelength_um)**2
    kz=np.sqrt(np.maximum(0,wavelength_um**-2-fx*fx-fy*fy))
    # Match the existing generator's float32 propagation phase arithmetic.
    transfer=np.asarray([pupil*np.exp(1j*np.float32(2*np.pi*z*kz)) for z in z_um],dtype=np.complex64)
    frame=0
    for seed in seeds:
        rng=np.random.RandomState(seed)
        for _ in range(frames_per_seed):
            phase=rng.rand(size,size)*2*np.pi
            field=np.pad(np.exp(1j*phase),((size//2,size//2),(size//2,size//2)))
            spectral=np.asarray(np.fft.fftshift(np.fft.fft2(field)),np.complex64)*pupil
            propagated=np.fft.ifft2(np.fft.ifftshift(spectral[None]*transfer,axes=(-2,-1)))
            crop=propagated[:,size//2:size//2+size,size//2:size//2+size]
            bank[frame]=np.abs(crop)**2
            frame+=1
        bank.flush()
    return {'path':str(path),'shape':list(shape),'seeds':list(seeds),'NA':na,
            'wavelength_um':wavelength_um,'pixel_um':pixel_um,'z_um':list(z_um),
            'phase_rng':'numpy RandomState, independent from acquisition',
            'normalization':'raw intensity, no per-frame normalization'}


class _Correlated(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,model):
        if g.shape[0]!=1:
            raise ValueError('Correlated variance expects one object per microbatch')
        bank=model.active_bank()
        n=len(bank);center=bank.mean(0,keepdim=True)
        predictions=[]
        for start in range(0,n,model.chunk):
            delta=(bank[start:start+model.chunk]-center)/model.sigma
            predictions.append(model.operator(g*delta[:,None]))
        y=torch.cat(predictions,0)
        ctx.model=model;ctx.bank=bank;ctx.center=center;ctx.save_for_backward(y)
        return y.square().sum(0,keepdim=True)/(n-1)

    @staticmethod
    def backward(ctx,grad):
        (y,)=ctx.saved_tensors;model=ctx.model;n=len(ctx.bank)
        dg=torch.zeros_like(ctx.bank[0:1,None])
        for start in range(0,n,model.chunk):
            delta=(ctx.bank[start:start+model.chunk]-ctx.center)/model.sigma
            back=model.operator.adjoint(2*y[start:start+model.chunk]*grad/(n-1))
            dg+=(back*delta[:,None]).sum(0,keepdim=True)
        return dg,None


class CorrelatedVariance(nn.Module):
    def __init__(self,operator,config):
        super().__init__();self.operator=operator;self.options=config['three_way']['covariance']
        self.chunk=int(self.options.get('chunk',8));self.evaluating=False
        self.sigma=float(self.options['sigma'])
        self._banks={}

    def active_bank(self):
        mode='validation' if self.evaluating else 'train'
        if mode not in self._banks:
            array=np.load(self.options[mode+'_bank'],mmap_mode='c')
            n=int(self.options['train_count']) if mode=='train' else len(array)
            self._banks[mode]=torch.as_tensor(np.array(array[:n]),device=self.operator.H.device)
        return self._banks[mode]

    def forward(self,g,measured_mean):
        return _Correlated.apply(g,self)
