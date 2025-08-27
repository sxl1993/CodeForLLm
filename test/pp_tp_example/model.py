
import torch
import torch.nn as nn
import torch.distributed as dist


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
