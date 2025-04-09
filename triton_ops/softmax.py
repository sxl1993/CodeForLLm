import os
# os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl
from triton.runtime import driver



@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)
    # print(f"row_start: {row_start}, row_step: {row_step}")
    # tl.device_print("num_stages", num_stages)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        tl.device_print("BLOCK_SIZE", BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        tl.device_print("row_minus_max", row_minus_max)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

@triton.jit
def online_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)
 
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        max_val = -float('inf')
        exp_sum = 0.0
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        current_max = tl.max(row, axis=0)
        # print(f"current_max: {current_max}")
        exp_sum = exp_sum * tl.math.exp2(max_val - current_max)
        data_exp = tl.math.exp2(row - current_max)
        current_exp_sum = tl.sum(data_exp, axis=0)
        exp_sum += current_exp_sum
        # print(f"data_exp: {data_exp}, exp_sum: {exp_sum}")
        output = data_exp / exp_sum
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x, dim=-1):
    assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]  # pre_dim
        inp = x.contiguous()
    if x.dtype is None:
        dtype = x.dtype
    
    K = inp.numel() // M // N  # post_dim
    print("K", K)
    exit()
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    # 每次循环迭代的块大小是大于 `x` 列数的最小二的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)


    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # 另一个技巧是通过增加每行分配的线程数来要求编译器使用更多的线程块 (`num_warps`)
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    # 将在下一个教程中看到如何以更自然的方式自动调整此值，以免自己进行手动启发式处理。
    num_warps = 8


    # Number of software piepling stages.
    # 软件流水线阶段的数量
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    # 分配输出空间
    y = torch.empty_like(x)
    
    # pre-compile kernel to get register usage and compute thread occupancy.
    # 预编译内核以获取寄存器使用情况并计算线程占用情况。
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = online_softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)


    num_programs = min(num_programs, n_rows)

    print(f"BLOCK_SIZE: {BLOCK_SIZE}, num_programs: {num_programs}, num_stages: {num_stages}, num_warps: {num_warps}")
    
    # Create a number of persistent programs.
    # 创建一些持久化程序。
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    
    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name` `x_name` 的不同可能值
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot 参数名，其值对应于图表中不同线条
        line_vals=['triton', 'torch'],  # possible values for `line_arg`` `line_arg` 的可能值
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # line styles 线条的样式
        ylabel="GB/s",  # label name for the y-axis y 轴的标签名称
        plot_name="online-softmax-performance",  # name for the plot. Used also as a file name for saving the plot. 图表的名称，也用作保存图表的文件名
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name` `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 128, 128, device='cuda')
    
    print(x.shape)
    output_triton = softmax(x, dim=-1)
    output_torch = torch.nn.functional.softmax(x, dim=1)
    if torch.allclose(output_torch, output_triton):
        print("torch and trion is equal")
    # benchmark.run(show_plots=True, print_data=True, save_path="./results")
    