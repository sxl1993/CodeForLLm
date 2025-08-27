import torch
import torch.nn as nn
import torch.distributed as dist
import os
# from model import TPLinear


def get_tp_pp_ranks(world_size, tp_size, pp_size):
    # TP groups
    tp_groups = []
    for i in range(world_size // tp_size):
        tp_groups.append(list(range(i * tp_size, (i + 1) * tp_size)))

    # PP groups
    pp_groups = []
    for i in range(world_size // pp_size):
        pp_groups.append(list(range(i, world_size, world_size // pp_size)))

    return tp_groups, pp_groups


def init_tp_pp_groups(rank, world_size, tp_size, pp_size):
    tp_groups, pp_groups = get_tp_pp_ranks(world_size, tp_size, pp_size)

    tp_group = None
    pp_group = None
    tp_rank = None
    pp_rank = None
    pp_ranks = None

    for group in tp_groups:
        if rank in group:
            tp_group = dist.new_group(ranks=group)
            tp_rank = group.index(rank)

    for group in pp_groups:
        if rank in group:
            pp_group = dist.new_group(ranks=group)
            pp_rank = group.index(rank)
            pp_ranks = group

    return tp_group, pp_group, tp_rank, pp_rank, pp_ranks

class TPLinear(nn.Module):
    """Linear layer split along output dim (Tensor Parallel)"""
    def __init__(self, input_size, output_size, tp_rank, tp_size, tp_group):
        super().__init__()
        assert output_size % tp_size == 0
        self.local_out = output_size // tp_size
        self.linear = nn.Linear(input_size, self.local_out)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group

    def forward(self, x):
        # print(f"Rank {self.tp_rank} {self.tp_size}, x: {x.shape}")
        out = self.linear(x)
        # print(f"Rank {self.tp_rank} {self.tp_size}, out: {out.shape}")
        # All-gather across TP ranks to get full output
        gathered = [torch.zeros_like(out) for _ in range(self.tp_size)]
        dist.all_gather(gathered, out, group=self.tp_group)
        gathered = torch.cat(gathered, dim=-1)
        # print(f"Rank {self.tp_rank} {self.tp_size}, gathered: {gathered.shape}")
        return gathered


class PPStage(nn.Module):
    def __init__(self, tp_rank, tp_size, tp_group):
        super().__init__()
        self.layer = TPLinear(16, 16, tp_rank, tp_size, tp_group)

    def forward(self, x):
        return self.layer(x)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 方案1：在init_process_group之前设置设备
    torch.cuda.set_device(local_rank)  # 使用local_rank而不是rank
    
    # 方案2：在init_process_group中明确指定device_id（推荐）
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}")
    )

    tp_size = 2
    pp_size = 4

    tp_group, pp_group, tp_rank, pp_rank, pp_ranks = init_tp_pp_groups(
        rank, world_size, tp_size, pp_size
    )

    model = PPStage(tp_rank=tp_rank, tp_size=tp_size, tp_group=tp_group).cuda()

    # Dummy input (only on first pipeline stage)
    x = torch.randn(4, 16).cuda() if pp_rank == 0 else torch.empty(4, 16).cuda()

    # Pipeline forward pass
    if pp_rank > 0:
        src = pp_ranks[pp_rank - 1]
        dist.recv(x, src=src)
        print(f"[Rank {rank}] recv from {src}")

    print(f"[Rank {rank}] running stage {pp_rank}, input shape: {x.shape}")
    x = model(x)
    print(f"[Rank {rank}] stage output: {x.shape}")

    if pp_rank < pp_size - 1:
        dst = pp_ranks[pp_rank + 1]
        dist.send(x, dst=dst)
        print(f"[Rank {rank}] send to {dst}")

    if pp_rank == pp_size - 1:
        print(f"[Rank {rank}] Final output: {x[0][:4]}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()