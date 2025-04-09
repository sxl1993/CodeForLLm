import os
# os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton
import triton.language as tl


@triton.jit
def maximum(a, b):
    return tl.where(a > b, a, b)

@triton.jit
def max_kernel_1():
    pass


@triton.jit
def max_kernel(input_ptr, max_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # print(f"pid: {pid}, N: {N}")
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + offset
    # print(f"offset: {offset}")
    mask = offset < M * N
    inp_val = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    # print(f"inp_val: {inp_val}")
    block_max = tl.reduce(inp_val, axis=0, combine_fn=maximum)
    # print(f"block_max: {block_max}")
    tl.store(max_ptr + pid, block_max)

def max(x):
    M, N = x.shape
    x = x.contiguous()
    device = x.device
    dtype = x.dtype
    x_flat = x.flatten()
    n_elements = x_flat.numel()
    BLOCK_SIZE = triton.next_power_of_2(N)
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    intermediate_max = torch.empty(n_blocks, device='cuda')
    # print(BLOCK_SIZE, n_blocks)
    # grid = lambda META: (M,)
    
    max_kernel[(M, 1, 1)](
        x,
        intermediate_max,
        M,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    final_max = intermediate_max.max()
    return final_max



if __name__ == "__main__":
    # torch.manual_seed(0)
    # device = torch.device("cuda:0")
    # x = torch.randn(4096, 2048).to(device)
    # # print(x)
    # torch_y = torch.max(x)
    # print(torch_y)
    # triton_y = max(x)
    # print(triton_y)
    benchmark.run(show_plots=True, print_data=True, save_path="./results")