"""Re-score completed diagonal candidates with unseen physical probes only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_cs_information import CachedCs, blur, load_operator, observations, vector_loss


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--config",default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--device",required=True)
    parser.add_argument("--run",required=True)
    parser.add_argument("--probes",type=int,default=2048)
    parser.add_argument("--bank",type=int,default=0)
    args=parser.parse_args()
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    dataset,directory=Path(args.dataset),Path(args.run)
    metadata=json.loads((dataset/"metadata.json").read_text())
    summary=json.loads((directory/"summary.json").read_text())
    if summary["history"][-1]["step"]!=199:
        raise RuntimeError("Only audit a completed 200-step run")
    operator=load_operator(Path(args.config).resolve(),device)
    cs=CachedCs(torch.from_numpy(np.load(dataset/"cs_kernel.npy")).to(device)*metadata["model_pattern_variance_mean"],(260,260))
    frames=np.load(dataset/"holdout_frames.npy",mmap_mode="r")
    target=blur(torch.from_numpy(frames.var(0,ddof=1)).to(device),0.5)
    mean=torch.from_numpy(frames.mean(0)).to(device)
    mask=mean>0.005*mean.max()
    candidates={"anchor":dataset/"anchor_rl3.npy","best":directory/"reconstruction_best.npy","final":directory/"reconstruction_final.npy"}
    report={"arguments":vars(args),"scope":"Unseen model probes; same empirical holdout frames, not a new acquisition test set","scores":{}}
    for name,path in candidates.items():
        shape=torch.from_numpy(np.load(path)[4]).to(device).clamp_min(0)
        shape/=shape.sum()
        samples=observations(shape,cs,operator,count=args.probes,batch=16,master=20260901,
                             domain="final_object",step=args.bank,sigma=0.5)
        prediction=samples.mean(0)
        mse,corr=vector_loss(prediction[mask][None],target[mask][None])
        # Covariance mean map MC uncertainty relative to its total signal energy.
        sem_energy=samples.var(0,unbiased=True).sum()/args.probes
        rel_sem=float((sem_energy/prediction.square().sum()).sqrt())
        report["scores"][name]={"holdout_loss":float(mse),"correlation":float(corr),"map_relative_mc_standard_error":rel_sem}
        print(json.dumps({"candidate":name,**report["scores"][name]}),flush=True)
    (directory/f"model_probe_audit_{args.probes}_bank{args.bank}.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
