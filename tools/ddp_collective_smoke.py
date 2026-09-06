from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=60),
        device_id=device,
    )
    value = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    print(
        f"rank={rank} local_rank={local_rank} physical={os.environ.get('SPECKLE_PHYSICAL_GPUS')} "
        f"sum={value.item():g}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
