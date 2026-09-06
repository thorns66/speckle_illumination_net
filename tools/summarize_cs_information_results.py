from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tools.summarize_nested_frame_count_cs import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default="outputs/linear_float_oracle_50um/cs_information")
    args=parser.parse_args()
    root=Path(args.root).resolve()
    rows=[]
    pending=[]
    for directory in sorted(root.glob("n*")):
        if not directory.is_dir():
            continue
        path=directory/"summary.json"
        if not path.exists():
            pending.append(directory.name)
            continue
        summary=json.loads(path.read_text())
        if summary["history"][-1]["step"]!=199:
            raise RuntimeError(f"Not a completed 200-step run: {directory}")
        dataset=Path(summary["arguments"]["dataset"])
        truth=np.load(dataset/"target.npy")
        anchor_metrics=evaluate(np.load(dataset/"anchor_rl3.npy"),truth)
        for checkpoint in ("best","final"):
            metrics=evaluate(np.load(directory/f"reconstruction_{checkpoint}.npy"),truth)
            row={"run":directory.name,"checkpoint":checkpoint,"frames":summary["frames"],
                 "step":summary["best_step"] if checkpoint=="best" else 199,
                 "holdout_loss":summary["best_holdout_loss"] if checkpoint=="best" else summary["history"][-1]["holdout_loss"],**metrics,
                 "ftc_change_percent":100*(metrics["ftc_um"]/anchor_metrics["ftc_um"]-1),
                 "truth_l2_change_percent":100*(metrics["unit_mass_relative_l2"]/anchor_metrics["unit_mass_relative_l2"]-1)}
            rows.append(row)
    report={"complete":not pending,"pending":pending,"selection":"independent holdout only; truth metrics diagnostic after completion",
            "evaluation_center":[10,12],"artifact_annulus":[20,120],"rows":rows}
    (root/"summary_latest.json").write_text(json.dumps(report,indent=2))
    if rows:
        with (root/"summary_latest.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    dataset=root.parent/"frame_count_nested/n800"
    truth=np.load(dataset/"target.npy")[4]
    images=[("Anchor",np.load(dataset/"anchor_rl3.npy")[4])]
    for title,name in (("Self 16","n800_self16_b0p5"),("Unbiased 16","n800_ustat16_b0p5"),("Unbiased 64","n800_ustat64_b0p5")):
        if (root/name/"summary.json").exists():
            images.append((title,np.load(root/name/"reconstruction_best.npy")[4]))
    figure,axes=plt.subplots(1,len(images),figsize=(4*len(images),4),squeeze=False,constrained_layout=True)
    for axis,(title,image) in zip(axes[0],images):
        image=image/image.sum()
        axis.imshow(image[:140,:140],cmap="gray",vmin=0,vmax=1.5*truth.max()/truth.sum())
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(root/"estimator_comparison_latest.png",dpi=150)
    plt.close(figure)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
