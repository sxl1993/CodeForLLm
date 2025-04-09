import os

# os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton
import triton.language as tl


@triton.jit
def moe_permute_row_map_kernel(
    sorted_row_id_ptr, row_id_map_ptr, num_rows, topK, num_out_tokens, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows * topK
    source_row = tl.load(sorted_row_id_ptr + offsets, mask=mask)
    source_token_id = source_row // topK
    source_topK_id = source_row % topK
    dst_index = source_topK_id * num_rows + source_token_id
    value = tl.where(offsets < num_out_tokens, offsets, -1)
    tl.store(row_id_map_ptr + dst_index, value, mask=mask)


def launch_moe_permute_row_map(sorted_row_id, topK, num_out_tokens):
    num_rows = sorted_row_id.shape[0] // topK
    row_id_map = torch.empty(topK * num_rows, dtype=torch.int32, device='cuda')

    grid = lambda meta: (triton.cdiv(sorted_row_id.numel(), meta['BLOCK_SIZE']),)
    moe_permute_row_map_kernel[grid](
        sorted_row_id,
        row_id_map,
        num_rows,
        topK,
        num_out_tokens,
        BLOCK_SIZE=256
    )

    return row_id_map

if __name__ == "__main__":
    device = torch.device("cuda:0")
    sorted_row_id = torch.tensor([3, 5, 1, 4, 2, 0]).to(device)
    topK = 2
    num_out_tokens = sorted_row_id.numel()

    row_id_map = launch_moe_permute_row_map(sorted_row_id, topK, num_out_tokens)
    print(row_id_map)

