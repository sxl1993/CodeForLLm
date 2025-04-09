import os
# os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton
import triton.language as tl
# from triton.runtime import driver


DEVICE = torch.device("cuda:0")
torch.manual_seed(0)

@triton.jit
def online_softmax_kernel_v1(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptr = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    max_val = -float('inf')
    exp_sum = 0.0
    row = tl.load(input_ptr, mask=mask, other=-float('inf'))
    current_max = tl.max(row, axis=0)
    print(f"current_max: {current_max}")
    exp_sum = exp_sum * tl.exp(max_val - current_max)
    data_exp = tl.exp(row - current_max)
    current_exp_sum = tl.sum(data_exp, axis=0)
    exp_sum += current_exp_sum
    print(f"data_exp: {data_exp}, exp_sum: {exp_sum}")
    output = data_exp / exp_sum
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cidv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


# @triton.autotune(
#     configs=[
#         triton.Config({'TILE_N': 64}, num_warps=2),
#         triton.Config({'TILE_N': 128}, num_warps=4),
#         triton.Config({'TILE_N': 256}, num_warps=8),
#         triton.Config({'TILE_N': 512}, num_warps=8),
#         triton.Config({'TILE_N': 1024}, num_warps=8),
#     ],
#     key=['N']
# )
@triton.jit
def online_softmax_kernel(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr):
    # 获取当前处理的行索引
    pid_m = tl.program_id(axis=0)
    # 第一阶段：计算全局最大值和指数和
    m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
    # print(f"m: {m}, z: {z}")
    
    input_ptr += pid_m * N
    output_ptr += pid_m * N

    # 第一次遍历：计算全局统计量
    previous_multiple = prev_multiple_of(N, TILE_N)
    # print(f"previous_multiple: {previous_multiple}")
    for start_n in range(0, previous_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        # print(f"n_offsets: {n_offsets}")
        # 加载当前块数据
        inp = tl.load(input_ptr + n_offsets)
        # print(f"inp: {inp}")
        
        # 更新最大值
        m_new = tl.maximum(m, inp)
        # print(f"m_new: {m_new}")
        all_neg_inf = m_new == float("-inf")
        # print(f"all_neg_inf: {all_neg_inf}")
        # 调整历史累加值
        z = tl.where(all_neg_inf, z, z * tl.math.exp2(m - m_new) + tl.math.exp2(inp - m_new))
        # print(f"z: {z}")
        m = m_new
        
    for start_n in range(previous_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
        m_new = tl.maximum(m, inp)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.math.exp(m - m_new) + tl.math.exp(inp - m_new))
        m = m_new
    m_reduced = tl.max(m, axis=0)
    z = tl.sum(z * tl.math.exp2(m - m_reduced), 0)
    m = m_reduced
    
    previous_multiple = prev_multiple_of(N, TILE_N)
    # specialize the first iteration
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(
            input_ptr + n_offsets,
            mask=mask,
            other=-float("inf"),
            eviction_policy="evict_first",
        )
        o = tl.math.exp2(inp - m) / z
        tl.store(output_ptr + n_offsets, o, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
        inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
        o = tl.exp(inp - m) / z
        tl.store(output_ptr + n_offsets, o)

# @triton.autotune(
#     configs=[
#         triton.Config({'TILE_N': 64}, num_warps=2),
#         triton.Config({'TILE_N': 128}, num_warps=4),
#         triton.Config({'TILE_N': 256}, num_warps=8),
#         triton.Config({'TILE_N': 512}, num_warps=8),
#         triton.Config({'TILE_N': 1024}, num_warps=8),
#     ],
#     key=['N']
# )
@triton.jit
def softmax_kernel(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * N
    col_offsets = tl.arange(0, TILE_N)
    
    input_ptr = row_start_ptr + col_offsets
    mask = col_offsets < N
    data = tl.load(input_ptr, mask=mask, other=-float('inf'))
    data_minus_max= data - tl.max(data, axis=0)
    data_exp = tl.math.exp2(data_minus_max)
    data_sum = tl.sum(data_exp, axis=0)
    softmax_output = data_exp / data_sum
    output_row_start_ptr = output_ptr + row_idx * N
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

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
    
    out = torch.empty_like(x)
    K = inp.numel() // M // N  # post_dim
    print(f"M:{M}, N:{N}, K: {K}")
    TILE_N = 1024
    # TILE_N = triton.next_power_of_2(N)
    grid = (M, 1, 1)
    online_softmax_kernel[grid](
        out,
        inp,
        M, 
        N,
        TILE_N=TILE_N
    )
    return out


if __name__ == "__main__":
    # x = torch.randn(2, 256, dtype=torch.float32).to(DEVICE)
    # # print(f"x: {x}\n")
    
    # output_triton = softmax(x)
    # # print(f"output_triton: {output_triton}")
    # output_torch = torch.nn.functional.softmax(x, dim=1)
    # # print(f"output_torch: {output_torch}")
    # if torch.allclose(output_torch, output_triton):
    #     print("torch and trion is equal")
    benchmark.run(show_plots=True, print_data=True, save_path="./results")