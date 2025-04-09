import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dims, num_heads, atten_dropout=0.1) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        assert hidden_dims // num_heads
        self.head_dims = hidden_dims // num_heads

        self.q_proj = nn.Linear(hidden_dims, hidden_dims)
        self.k_proj = nn.Linear(hidden_dims, hidden_dims)
        self.v_proj = nn.Linear(hidden_dims, hidden_dims)
        self.output_proj = nn.Linear(hidden_dims, hidden_dims)
        self.atten_dropout = nn.Dropout(atten_dropout)

    def forward(self, x, atten_mask=None):
        bs, seq, _ = x.size() 
        q = self.q_proj(x) # (bs, seq, hidden_dims)
        k = self.k_proj(x) # (bs, seq, hidden_dims)
        v = self.v_proj(x) # (bs, seq, hidden_dims)

        q = q.view(bs, seq, self.num_heads, self.head_dims).permute(0, 2, 1, 3) # (bs, num_heads, seq, head_dims)
        k = k.view(bs, seq, self.num_heads, self.head_dims).permute(0, 2, 1, 3) # (bs, num_heads, seq, head_dims)
        v = v.view(bs, seq, self.num_heads, self.head_dims).permute(0, 2, 1, 3) # (bs, num_heads, seq, head_dims)

        atten_weight = torch.matmul(q, k.transpose(2, 3) / math.sqrt(self.head_dims)) # (bs, num_heads, seq, seq)
        if atten_mask is not None:
            atten_mask = atten_mask.masked_fill(
                atten_mask ==0, float=('-inf')
            )
        atten_weight = torch.softmax(atten_weight, dim=-1)
        atten_weight = self.atten_dropout(atten_weight)

        output = torch.matmul(atten_weight, v) # (bs, num_heads, seq, head_dims)
        output = output.permute(0, 2, 1, 3).reshape(bs, seq, -1) # (bs, seq, num_heads, head_dims) -> (bs, seq, num_heads*head_dims)
        output = self.output_proj(output) 
        return output

if __name__ == "__main__":
    batch_size = 128
    seq_length = 64
    hidden_dims = 512
    num_heads = 8
    model = MultiHeadAttention(hidden_dims, num_heads)
    x = torch.randn(batch_size, seq_length, hidden_dims)
    output = model(x)
    print(output.shape)
        




        