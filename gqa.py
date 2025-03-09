import math
import torch
import torch.nn as nn

torch.manual_seed(0)

class GroupQueryAtention(nn.Module):
    def __init__(self, hidden_dims, query_heads, key_value_heads, atten_dropout=0.1) -> None:
        super().__init__()
        self.query_heads = query_heads
        self.key_value_heads = key_value_heads
        self.atten_dropout = atten_dropout

        self.head_dims = hidden_dims // query_heads
        self.num_head_group = query_heads // key_value_heads
        self.q_proj = nn.Linear(hidden_dims, query_heads * self.head_dims)
        self.k_proj = nn.Linear(hidden_dims, key_value_heads * self.head_dims)
        self.v_proj = nn.Linear(hidden_dims, key_value_heads * self.head_dims)
        self.output_proj = nn.Linear(hidden_dims, hidden_dims)
        self.atten_dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        bs, seq, _ = q.size()
        q = q.view(bs, seq, self.query_heads, self.head_dims).permute(0, 2, 1, 3)     # (bs, seq, query_heads, head_dims) => (bs, num_query_heads, seq, head_dims)
        k = k.view(bs, seq, self.key_value_heads, self.head_dims).permute(0, 2, 1, 3) # (bs, seq, key_value_heads, head_dims) => (bs, key_value_heads, seq, head_dims)
        v = v.view(bs, seq, self.key_value_heads, self.head_dims).permute(0, 2, 1, 3) # (bs, seq, key_value_heads, head_dims) => (bs, key_value_heads, seq, head_dims)

        # (bs, num_query_heads, seq, head_dims) => (bs, num_head_group, key_value_heads, seq, head_dims)
        q = q.view(bs, self.num_head_group, -1, seq, self.head_dims) 
        # (bs, key_value_heads, seq, head_dims) => (bs, 1, key_value_heads, seq, head_dims) => (bs, 1, key_value_heads, head_dims, seq)
        k = k.unsqueeze(1).permute(0, 1, 2, 4, 3) 
        # (bs, key_value_heads, seq, head_dims) => (bs, 1, key_value_heads, seq, head_dims)
        v = v.unsqueeze(1)
        attn_weight = torch.matmul(q, k) # (bs, num_head_group, key_value_heads, seq, seq)

        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.atten_dropout(attn_weight) # (bs, num_head_group, key_value_heads, seq, seq)
        output = torch.matmul(attn_weight, v) # (bs, num_head_group, key_value_heads, seq, head_dims)
        # (bs, num_head_group, key_value_heads, seq, head_dims) => (bs, seq, num_head_group, key_value_heads, head_dims) => (bs, seq, num_head_group*key_value_heads*head_dims)
        output = output.permute(0, 3, 1, 2, 4).reshape(bs, seq, -1) 
        output = self.output_proj(output)
        # print(output.shape)
        return output


if __name__ == "__main__":
    batch_size = 128
    seq_length = 64
    hidden_dims = 512
    query_heads = 8
    key_value_heads = 2
    model = GroupQueryAtention(hidden_dims, query_heads, key_value_heads)
    x = torch.randn(batch_size, seq_length, hidden_dims)
    output = model(x, x, x)
    print(output.shape)



