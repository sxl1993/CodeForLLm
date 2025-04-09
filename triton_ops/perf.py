import torch
import triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 4)],  # different possible values for `x_name` `x_name` 的不同可能值
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot 参数名，其值对应于图表中不同线条
        line_vals=['triton', 'torch'],  # possible values for `line_arg`` `line_arg` 的可能值
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # line styles 线条的样式
        ylabel="GB/s",  # label name for the y-axis y 轴的标签名称
        plot_name="max-performance",  # name for the plot. Used also as a file name for saving the plot. 图表的名称，也用作保存图表的文件名
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name` `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, provider, op):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    # print(f"x shape: {x.shape}")
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.max(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: max(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)