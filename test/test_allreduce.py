# all_reduce_example.py
import os
import torch
import torch.distributed as dist

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")

    # 获取当前进程的 rank 和设备
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # 设置当前进程的 CUDA 设备
    torch.cuda.set_device(local_rank)

    # 构造一个初始值为 rank 的 tensor
    tensor = torch.tensor([float(rank)], device=torch.cuda.current_device())
    print(f"[Before All-Reduce] Rank {rank}: {tensor.item()}")

    # 执行 all_reduce（求和）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"[After  All-Reduce] Rank {rank}: {tensor.item()}")

    # 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
