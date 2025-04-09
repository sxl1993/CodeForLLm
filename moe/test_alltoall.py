import os
import torch
import torch.distributed as dist

def all_to_all_example(tokens_per_expert, expert_num, world_size, rank):
    local_expert_num = expert_num // world_size
    input_split_sizes = [local_expert_num] * world_size
    output_split_sizes = [local_expert_num] * world_size
    parallel_tokens_per_expert = torch.empty_like(tokens_per_expert, dtype=torch.int32).to(torch.device(rank))
    dist.all_to_all_single(parallel_tokens_per_expert, tokens_per_expert, output_split_sizes, input_split_sizes)
    return parallel_tokens_per_expert

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
    global_input_tokens = all_to_all_example(tokens_per_expert, expert_num, world_size, rank)
    global_input_tokens = global_input_tokens.view(world_size, num_local_experts)
    print(global_input_tokens)

    dist.destroy_process_group()