"""Read-only-on-data model audit: stationary Cs versus finite-generator ensemble."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from physics.speckle_oracle import SpeckleGeneratorConfig, circular_pupil
from tools.diagnose_cs_information import CachedCs, load_operator, probe_seed, _lowpass_probes, vector_loss


@torch.no_grad()
def raw_patterns(count, device, seed):
    config = SpeckleGeneratorConfig(seed=seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    pupil = torch.fft.ifftshift(circular_pupil(config, device=device))
    size, pad = config.sampling, config.sampling // 2
    outputs = []
    for start in range(0, count, 32):
        phase = torch.rand((min(32, count-start), size, size), device=device, generator=generator) * (2*torch.pi)
        source = F.pad(torch.polar(torch.ones_like(phase), phase), (pad,pad,pad,pad))
        field = torch.fft.ifft2(torch.fft.fft2(source) * pupil)
        outputs.append(field.abs().square()[:, pad:pad+size, pad:pad+size].cpu())
    return torch.cat(outputs)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patterns", type=int, default=4096)
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260971)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dataset, output = Path(args.dataset), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    operator = load_operator(Path(args.config).resolve(), device)
    metadata = json.loads((dataset/"metadata.json").read_text())
    kernel = torch.from_numpy(np.load(dataset/"cs_kernel.npy")).to(device)*metadata["model_pattern_variance_mean"]
    cs = CachedCs(kernel, (260,260))
    q = torch.from_numpy(_lowpass_probes(args.probes,(260,260),sigma=8,
                        seed=probe_seed(20260901,"final_sensor",0,0)%(2**32))).to(device)
    back = torch.cat([operator.adjoint(q[i:i+4,None]) for i in range(0,len(q),4)])
    targets = {}
    for split in ("train", "holdout"):
        frames = torch.from_numpy(np.load(dataset/f"{split}_frames.npy")).to(device).flatten(1)
        frames = frames-frames.mean(0,keepdim=True)
        targets[split] = ((q.flatten(1) @ frames.T) @ frames / (len(frames)-1)).reshape_as(q)
    del frames
    patterns = raw_patterns(args.patterns,device,args.seed).to(device).flatten(1)/metadata["speckle_system_mean_before_scaling"]
    patterns = patterns-patterns.mean(0,keepdim=True)

    root = dataset.parent.parent/"cs_information"
    candidates = {"truth":dataset/"target.npy", "anchor":dataset/"anchor_rl3.npy",
                  "vector_final":root/"n800_vector_b0p5/reconstruction_final.npy",
                  "diagonal_old_best":dataset.parent.parent/"frame_count_results/n800_b0p5_s0p5/reconstruction_best.npy"}
    report = {"arguments":vars(args),"scores":{},"scope":"Same sensor probes, fixed system gain, no reconstruction or free scale fitting"}
    for name,path in candidates.items():
        g=torch.from_numpy(np.load(path)[4]).to(device).clamp_min(0)
        g=g/g.sum()
        predictions={"stationary":[],"finite_ensemble":[]}
        for start in range(0,len(q),4):
            right=g[None]*back[start:start+4,0,0]
            finite=((right.flatten(1)@patterns.T)@patterns/(len(patterns)-1)).reshape_as(right)
            for kind,correlated in (("stationary",cs.action(right)),("finite_ensemble",finite)):
                predictions[kind].append(operator((g*correlated)[:,None,None])[:,0])
        scores={}
        for kind,parts in predictions.items():
            prediction=torch.cat(parts)
            scores[kind]={}
            for split,target in targets.items():
                per,corr=vector_loss(prediction,target)
                global_mse=(prediction-target).square().sum()/target.square().sum()
                scores[kind][split]={"per_probe_mse":float(per),"global_mse":float(global_mse),"correlation":float(corr)}
        report["scores"][name]=scores
        print(json.dumps({"candidate":name,"scores":scores}),flush=True)
    output.write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
