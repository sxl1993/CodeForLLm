import os
import torch
import torch.distributed as dist

def allgather_example(tokens_per_expert, expert_num, world_size, rank):
    num_local_experts = expert_num // world_size
    device = torch.device(f'cuda:{rank}')
    tensor_out = torch.zeros(world_size, expert_num, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(tensor_out, tokens_per_expert)
    local_expert_indices_offset = (rank * num_local_experts)
    local_expert_indices = [local_expert_indices_offset + i for i in range(num_local_experts)]
    return tensor_out[:, local_expert_indices]

if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    tokens_per_expert = torch.tensor([
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 3, 0, 0, 0, 
    5, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 5, 
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2], dtype=torch.int32).to(torch.device(rank))
    expert_num = tokens_per_expert.size(0)
    num_local_experts = expert_num // world_size
    local_input_tokens = allgather_example(tokens_per_expert, expert_num, world_size, rank)
    print(local_input_tokens)
    dist.destroy_process_group()