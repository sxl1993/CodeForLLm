import random
import numpy as np
import torch
import flashinfer
from typing import Optional, List

import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.set_default_device("cuda")
seed = 42
torch.manual_seed(42)
random.seed(seed)
np.random.seed(seed)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        print(q.shape, k.shape, v.shape)
        attn = torch.matmul(q, k.transpose(1, 2)).float()
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.matmul(attn, v).permute(1, 0, 2)
        outputs.append(out)
        start_idx += query_len

    output = torch.cat(outputs, dim=0)
    return output


def extract_kv_vectorized_v1_optimized(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: List[int],
    block_size: int,
    num_heads: int,
    head_size: int,
    num_seqs: int,
):
    # Step1: 拼接所有 key/value 序列对应的块，构造一个大 kv 张量
    kv_outputs = []
    kv_lens_cumsum = [0] + torch.cumsum(torch.tensor(kv_lens), dim=0).tolist()

    # 把 key/value按序列展开，拼接成大的key/value
    all_k_list = []
    all_v_list = []
    for i in range(num_seqs):
        num_kv_blocks = (kv_lens[i] + block_size - 1) // block_size
        block_idx = block_tables[i, :num_kv_blocks]
        # (kv_len_i, num_heads, head_size)
        k_seq = key_cache[block_idx].view(-1, num_heads, head_size)[: kv_lens[i]]
        v_seq = value_cache[block_idx].view(-1, num_heads, head_size)[: kv_lens[i]]
        all_k_list.append(k_seq)
        all_v_list.append(v_seq)

    # (sum_kv_len, num_heads, head_size)
    key_all = torch.cat(all_k_list, dim=0)
    # (sum_kv_len, num_heads, head_size)
    value_all = torch.cat(all_v_list, dim=0)
    return key_all, value_all


def attention_mask_vectorized_v1_optimized(
    query_lens: torch.Tensor, kv_lens: torch.Tensor, attn: torch.Tensor, num_seqs: int
) -> torch.Tensor:
    # 准备索引，batch_offsets用来拆分输出
    query_offsets = [0] + torch.cumsum(query_lens, dim=0).tolist()
    kv_offsets = [0] + torch.cumsum(kv_lens, dim=0).tolist()

    # 构造mask，屏蔽不同序列无效的 key位置
    mask = torch.zeros(
        (sum(query_lens), sum(kv_lens)), dtype=torch.bool, device=attn.device
    )
    for i in range(num_seqs):
        q_start, q_end = query_offsets[i], query_offsets[i + 1]
        k_start, k_end = kv_offsets[i], kv_offsets[i + 1]
        # 这块是有效区域，其他置True（mask掉）
        mask[q_start:q_end, :k_start] = True
        mask[q_start:q_end, k_end:] = True

    # attn的第二维是query维，第三维是key维，mask维度是 (Q_total, K_total)，需要扩展到 (head, Q_total, K_total)
    attn_mask = mask.unsqueeze(0).expand(attn.shape[0], -1, -1)
    attn.masked_fill_(attn_mask, float("-inf"))
    return attn


def attention_mask_vectorized_v2_optimized(
    query_lens: torch.Tensor, kv_lens: torch.Tensor, attn: torch.Tensor
) -> torch.Tensor:
    # 获取每个 query token 和 kv token 所属的序列 id
    q_seq_ids = torch.repeat_interleave(
        torch.arange(len(query_lens), device=query_lens.device), query_lens
    )
    k_seq_ids = torch.repeat_interleave(
        torch.arange(len(kv_lens), device=kv_lens.device), kv_lens
    )

    # 构造 (Q_total, K_total) 的 mask: 如果不属于同一序列，置 True
    mask = (
        q_seq_ids[:, None] != k_seq_ids[None, :]
    )  # broadcasting 得到 [Q_total, K_total] 的 bool 矩阵

    # 扩展到 (head, Q_total, K_total)
    attn_mask = mask.unsqueeze(0).expand(attn.shape[0], -1, -1)

    # 应用 mask
    attn.masked_fill_(attn_mask, float("-inf"))
    return attn


def extract_kv_vectorized_v4_optimized(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    block_size: int,
):
    def get_positions_in_seq(kv_lens: torch.Tensor) -> torch.Tensor:
        total_tokens = kv_lens.sum()
        seq_offsets = torch.cumsum(kv_lens, 0) - kv_lens
        return torch.arange(
            total_tokens, device=kv_lens.device
        ) - torch.repeat_interleave(seq_offsets, kv_lens)

    device = key_cache.device

    # 1. 构造 seq_ids: 每个 token 属于哪个序列
    seq_ids = torch.repeat_interleave(
        torch.arange(len(kv_lens), device=device), kv_lens
    )

    # 2. 构造 positions_in_seq: 每个 token 在所属序列中的偏移
    positions_in_seq = get_positions_in_seq(kv_lens)

    # 3. 计算每个 token 对应的块信息
    local_block_ids = (
        positions_in_seq // block_size
    )  # 每个 token 在所属序列中的第几个 block
    pos_in_block = positions_in_seq % block_size  # token 在当前 block 中的偏移位置

    # 4. 查表：根据 seq_id 和 local_block_id 找到全局 block ID
    global_block_ids = block_tables[seq_ids, local_block_ids]

    # 5. 一次性从 cache 中取出 token 对应的 KV 向量
    key_all = key_cache[
        global_block_ids, pos_in_block
    ]  # shape: [total_tokens, num_head, head_dim]
    value_all = value_cache[global_block_ids, pos_in_block]
    return key_all, value_all


def packed_paged_attention(
    query: torch.Tensor,  # (sum_query_len, num_heads, head_size)
    key_cache: torch.Tensor,  # (num_blocks, block_size, num_heads, head_size)
    value_cache: torch.Tensor,  # (num_blocks, block_size, num_heads, head_size)
    query_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    block_tables: torch.Tensor,  # (num_seqs, max_num_blocks)
    scale: float,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    _, block_size, num_heads, head_size = key_cache.shape

    key_all, value_all = extract_kv_vectorized_v4_optimized(
        key_cache, value_cache, block_tables, kv_lens, block_size
    )

    # Step3: permute 方便计算 (seq_len, num_heads, head_size) -> (num_heads, seq_len, head_size)
    q = query.permute(1, 0, 2)  # (num_heads, sum_query_len, head_size)
    k = key_all.permute(1, 0, 2)  # (num_heads, sum_kv_len, head_size)
    v = value_all.permute(1, 0, 2)  # (num_heads, sum_kv_len, head_size)
    print(q.shape, k.shape, v.shape)

    # Step4: scaled dot-product attention，先算完整的attn矩阵 (num_heads, sum_query_len, sum_kv_len)
    attn = torch.matmul(q, k.transpose(1, 2)).float()
    attn *= scale

    attn = attention_mask_vectorized_v2_optimized(query_lens, kv_lens, attn)

    # Step6: softmax + 可选soft_cap
    if soft_cap is not None:
        attn = soft_cap * torch.tanh(attn / soft_cap)
    attn = torch.softmax(attn, dim=-1).to(v.dtype)

    # Step7: 计算输出 (num_heads, sum_query_len, head_size)
    out = torch.matmul(attn, v)

    # Step8: 转回 (sum_query_len, num_heads, head_size)
    out = out.permute(1, 0, 2)

    return out


@torch.inference_mode()
def test_flashinfer_prefill_with_paged_kv():
    seq_lens = [(1, 1358), (5, 18), (129, 463)]
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    query_lens = torch.tensor(query_lens)
    kv_lens = torch.tensor(kv_lens)
    num_query_heads = 16
    num_kv_heads = 16
    head_size = 128
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    dtype = torch.float16
    num_blocks = 1000
    block_size = 16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(
        num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )

    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=None,
    )
    output = wrapper.run(
        query,
        key_value_cache,
    )

    # ref_output = ref_paged_attn(query=query,
    #                             key_cache=key_cache,
    #                             value_cache=value_cache,
    #                             query_lens=query_lens,
    #                             kv_lens=kv_lens,
    #                             block_tables=block_tables,
    #                             scale=scale,
    #                             soft_cap=None)
    ref_output = packed_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=None,
    )

    torch.testing.assert_close(
        output, ref_output, atol=5e-2, rtol=1e-2
    ), f"{torch.max(torch.abs(output - ref_output))}"


if __name__ == "__main__":
    test_flashinfer_prefill_with_paged_kv()
