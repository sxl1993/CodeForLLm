import os

# os.environ["TRITON_INTERPRET"] = "1"

import triton.language as tl
import triton
import numpy as np
import random
import torch



torch.set_default_device("cuda")
seed = 42
torch.manual_seed(42)
random.seed(seed)
np.random.seed(seed)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs: list[torch.Tensor] = []

    for i in range(num_seqs):
        q = query[query_indptr[i]: query_indptr[i + 1]]
        q_scale = q * scale
        # print(f"q: {q_scale}")
        # print(f"q_scale: {q_scale[0][0]}")
        kv_len = kv_indptr[i + 1] - kv_indptr[i]
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        # print(f"i: {i}, block_indices: {block_indices}, {kv_len}, {key_cache[block_indices].shape}")

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        # if i == 0:
        #     print(f"k: {k.shape}, kv_len: {kv_len}")
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]
        q_scale = q_scale.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)             # (h, s, d)
        
        logits = torch.matmul(q_scale, k.transpose(1, 2)).float()
        attn = torch.softmax(logits, dim=-1).to(v.dtype)
        out = torch.matmul(attn, v).permute(1, 0, 2)
        # if i == 0:
            # print(f"kv_len: {kv_len}, block_indices: {block_indices}, q_scale: {q_scale}, {q_scale.shape}")
            # print(f"block_indices: {block_indices}, k: {k}, {k.shape}")
            # print(f"block_indices: {block_indices}, logits: {logits}")
            # print(f"block_indices: {block_indices}, attn: {attn}, {attn.shape}")   # (h, s1, s2)
            # print(f"v: {v}")                                                       # (h, s2, d)
            # print(f"block_indices: {block_indices}, out: {out}")                                                   # (h, s1, d)
           
        outputs.append(out)

    output = torch.cat(outputs, dim=0)
    return output


@triton.jit
def paged_attn_kernel(q_ptr, k_ptr, v_ptr, output_ptr,
                      q_indptr, kv_indptr, block_tables_ptr,
                      BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
                      HEAD_SIZE: tl.constexpr, scale: tl.constexpr,
                      num_q_heads: tl.constexpr, num_kv_heads: tl.constexpr, bs: tl.constexpr,
                      max_num_blocks_per_seq: tl.constexpr):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    if seq_id >= bs or head_id >= num_q_heads:
        return

    q_start = tl.load(q_indptr + seq_id)
    q_end = tl.load(q_indptr + seq_id + 1)
    kv_start = tl.load(kv_indptr + seq_id)
    kv_end = tl.load(kv_indptr + seq_id + 1)

    q_len = q_end - q_start
    kv_len = kv_end - kv_start

    num_kv_blocks = (kv_len + BLOCK_KV - 1) // BLOCK_KV
    q_offset = tl.arange(0, BLOCK_Q)
    k_offset = tl.arange(0, BLOCK_KV)
    d_offset = tl.arange(0, HEAD_SIZE)

    for q_id in range(0, q_len, BLOCK_Q):
        q_mask = q_offset + q_id < q_len
        q_offsets = (q_start + q_id + q_offset[:, None]) * num_q_heads * HEAD_SIZE + head_id * HEAD_SIZE + d_offset[None, :]
        q = tl.load(q_ptr + q_offsets, mask=q_mask[:, None], other=0)
        q *= scale
        # if seq_id == 0:
        #     print(f"seq_id: {seq_id}, head_id: {head_id}, {q}")
        max_logit = tl.full((BLOCK_Q, ), float('-inf'), dtype=tl.float32)
        denom = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, HEAD_SIZE), dtype=tl.float32)

        for b_id in range(0, num_kv_blocks):
            b_offset = seq_id * max_num_blocks_per_seq + b_id
            block_idx = tl.load(block_tables_ptr + b_offset)

            base_offset = block_idx * BLOCK_KV * num_kv_heads * HEAD_SIZE
            block_offset = k_offset[:, None] * num_kv_heads * HEAD_SIZE
            head_offset = head_id * HEAD_SIZE
            kv_offsets = base_offset + block_offset + head_offset + d_offset[None, :]
            block_token_start = b_id * BLOCK_KV
            block_token_len = min(kv_len - block_token_start, BLOCK_KV)
            kv_mask = k_offset < block_token_len
            
            k = tl.load(k_ptr + kv_offsets, mask=kv_mask[:, None], other=0)
            # if seq_id == 0:
            #     print(f"seq_id: {seq_id}, head_id: {head_id}, block_idx: {block_idx}, kv_len: {kv_len}, k: {k}")
            v = tl.load(v_ptr + kv_offsets, mask=kv_mask[:, None], other=0)

            logits_mask = q_mask[:, None] & kv_mask[None, :]
            logits = tl.dot(q, tl.trans(k)).to(tl.float32)
            
            logits = tl.where(logits_mask, logits, float('-inf'))
            # if seq_id == 0:
            #     print(f"seq_id: {seq_id}, head_id: {head_id}, block_idx: {block_idx}, logits: {logits}")

            current_max = tl.max(tl.where(logits_mask, logits, float('-inf')), axis=1)
            new_max = tl.maximum(current_max, max_logit)
            exp_logit = tl.exp(logits - new_max[:, None])
            exp_logit = tl.where(logits_mask, exp_logit, 0.0)
            scaled_old = tl.exp(max_logit - new_max)[:, None] * denom[:, None]
            denom = tl.sum(exp_logit, axis=1) + tl.sum(scaled_old, axis=1)
            max_logit = new_max

            # if seq_id == 1:
            #     print(f"seq_id: {seq_id}, head_id: {head_id}, block_idx: {block_idx}, attn: {attn}")
            acc = acc * (scaled_old / denom[:, None]) + tl.dot(exp_logit, v) / denom[:, None]

            # if seq_id == 0:
            #     print(f"seq_id: {seq_id}, head_id:{head_id}, block_idx: {block_idx}, acc: {acc}")
        out_off = (q_start + q_id + q_offset[:, None]) * num_q_heads * HEAD_SIZE + head_id * HEAD_SIZE + d_offset[None, :]
        tl.store(output_ptr + out_off, acc, mask=q_mask[:, None])



def paged_attention(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.tensor,
        block_tables: torch.Tensor,
        scale: float):
    _, num_q_heads, head_size = query.shape
    num_blocks, block_size, num_kv_heads, _ = key_cache.shape
    bs = qo_indptr.shape[0] - 1
    grid = (bs, num_q_heads)
    # print(f"grid: {grid}")

    output = torch.empty_like(query)

    paged_attn_kernel[grid](
        query,
        key_cache.contiguous(),
        value_cache.contiguous(),
        output,
        qo_indptr,
        kv_indptr,
        block_tables,
        BLOCK_Q=block_size,
        BLOCK_KV=block_size,
        HEAD_SIZE=head_size,
        scale=scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        bs=bs,
        max_num_blocks_per_seq=block_tables.shape[-1]
    )
    return output


if __name__ == "__main__":
    seq_lens = [(1, 1358), (5, 18), (129, 463)]
    # seq_lens = [(1, 4), (2, 16), (1, 16)]
    # seq_lens = [(1, 18)]
    max_bs = len(seq_lens)
    num_query_heads = 16
    num_kv_heads = 16
    head_size = 256

    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    query_lens = torch.tensor(query_lens)
    qo_indptr = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=query_lens.device),
        torch.cumsum(query_lens, dim=0, dtype=torch.int32)])

    kv_lens = torch.tensor(kv_lens)
    kv_indptr = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=kv_lens.device),
        torch.cumsum(kv_lens, dim=0, dtype=torch.int32)])

    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    dtype = torch.float32
    num_blocks = 10
    block_size = 16
    query = torch.randn(sum(query_lens), num_query_heads,
                        head_size, dtype=dtype)

    key_value_cache = torch.randn(
        num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )

    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    # print(key_cache.shape, query.shape)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (max_bs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # print("query:", query.shape, query)
    # print(f"key:", key_cache.shape)
    # key_cache = key_cache.contiguous()
    # t = key_cache.contiguous().view(-1)
    # print(t[64:68])
    # print(f"block_tables:", block_tables)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        block_tables=block_tables,
        scale=scale,
    )
    # print(ref_output[0][0])

    output = paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        block_tables=block_tables,
        scale=scale,
    )

    # print(output[0][0])

    torch.testing.assert_close(
        output, ref_output, atol=1e-3, rtol=1e-3
    ), f"{torch.max(torch.abs(output - ref_output))}"
