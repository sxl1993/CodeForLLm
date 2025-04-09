import torch
import torch.nn as nn
import torch.nn.functional as F
from grouped_gemm.ops import gmm

torch.manual_seed(0)

class SwiGLUActivatition(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        input = torch.chunk(input, 2, dim=-1)
        return F.silu(input[0]) * input[1]

class Expert(nn.Module):
    """
    Expert 模块，封装了 MoE 模型中单个专家的计算逻辑。
    每个专家是一个 MLP，包含两个线性变换和一个激活函数。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation_fn=nn.ReLU()):
        """
        初始化 Expert 模块。

        参数:
        - input_dim: 输入特征的维度
        - hidden_dim: 隐藏层的维度
        - output_dim: 输出特征的维度
        - activation_fn: 激活函数（默认使用 ReLU）
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn

        # 定义两个线性变换层
        self.linear1 = nn.Linear(input_dim, hidden_dim)  # 第一个线性变换
        self.linear2 = nn.Linear(hidden_dim, output_dim)  # 第二个线性变换

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
        - x: 输入张量，形状为 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)

        返回:
        - output: 输出张量，形状与输入张量相同（除了最后一个维度变为 output_dim）
        """
        # 第一个线性变换
        intermediate = self.linear1(x)  # 形状: (batch_size, seq_len, hidden_dim) 或 (batch_size, hidden_dim)
        
        # 应用激活函数
        activated = self.activation_fn(intermediate)
        
        # 第二个线性变换
        output = self.linear2(activated)  # 形状: (batch_size, seq_len, output_dim) 或 (batch_size, output_dim)
        
        return output


class Experts(nn.Module):
    """
    封装多个 Expert 模块，支持批量计算。
    """

    def __init__(self, 
                 num_experts: int, 
                 expert_intermediate_size: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 w1: torch.Tensor, 
                 w2: torch.Tensor,
                 activation_fn=SwiGLUActivatition(),
                 ) -> None:
        """
        初始化 Experts 模块。

        参数:
        - num_experts: 专家数量
        - input_dim: 输入特征的维度
        - hidden_dim: 隐藏层的维度
        - output_dim: 输出特征的维度
        - activation_fn: 激活函数（默认使用 ReLU）
        """
        super().__init__()
        self.num_experts = num_experts
        self.expert_intermediate_size = expert_intermediate_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.w1 = w1
        self.w2 = w2
        self.activation_fn = activation_fn

    def forward(self, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
        - x: 输入张量，形状为 (batch_size, seq_len, input_dim)
        - expert_indices: 每个 token 分配的专家索引，形状为 (batch_size, seq_len)

        返回:
        - output: 输出张量，形状为 (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = hidden_states.shape
        topk = tokens_per_expert.size(-1)
        hidden_states = hidden_states.view(-1, input_dim)  # 将输入展平为 (batch_size * seq_len, input_dim)
        tokens_per_expert = tokens_per_expert.view(-1, topk)
        flatten_tokens_per_expert = tokens_per_expert.view(-1)
        sorted_tokens_per_expert = torch.argsort(flatten_tokens_per_expert, stable=True)
        permuted_tokens = hidden_states.index_select(0, sorted_tokens_per_expert // topk)

        tokens_per_expert = torch.histc(tokens_per_expert, bins=self.num_experts, min=0, max=self.num_experts)
        tokens_per_expert = tokens_per_expert.to(torch.device('cpu')).to(torch.int64)
        # print(tokens_per_expert)
        # 初始化输出
        output = torch.zeros_like(permuted_tokens, device=hidden_states.device, dtype=torch.bfloat16)

        # 遍历每个专家
        start_idx = 0
        for expert_idx in range(self.num_experts):
            # 获取当前专家分配的 token 数量
            num_tokens = tokens_per_expert[expert_idx].item()

            if num_tokens == 0:
                continue  # 如果没有 token 分配给当前专家，跳过

            # 获取当前专家分配的输入数据
            expert_input = permuted_tokens[start_idx:start_idx + num_tokens]  # 形状: (num_tokens, hidden_dim)

            # 第一个线性变换: expert_input @ w1[expert_idx]
            intermediate = torch.matmul(expert_input.to(torch.bfloat16), self.w1[expert_idx])  # 形状: (num_tokens, expert_intermediate_size * 2)
            # 应用激活函数
            intermediate = self.activation_fn(intermediate)

            # 第二个线性变换: intermediate @ w2[expert_idx]
            expert_output = torch.matmul(intermediate.to(torch.bfloat16), self.w2[expert_idx])  # 形状: (num_tokens, output_dim)
            # 将当前专家的输出写入最终输出
            output[start_idx:start_idx + num_tokens] = expert_output

            # 更新起始索引
            start_idx += num_tokens

        # 将输出恢复为原始形状
        # output = output.view(batch_size, seq_len, self.output_dim)
        return output


class GroupedGemmExperts(nn.Module):
    def __init__(self,
                 num_experts: int,
                 expert_intermediate_size: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 w1: torch.Tensor,
                 w2: torch.Tensor,
                 activation_fn=SwiGLUActivatition()) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.expert_intermediate_size = expert_intermediate_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.w1 = w1
        self.w2 = w2
        self.activation_fn = activation_fn

    def forward(self, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
        - hidden_states: 输入张量，形状为 (batch_size, seq_len, input_dim)
        - tokens_per_expert: 每个专家分配的token数, 形状为(expert_nums)

        返回:
        - output: 输出张量，形状为 (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = hidden_states.shape
        topk = tokens_per_expert.size(-1)
        hidden_states = hidden_states.view(-1, input_dim)    # 将输入展平为 (batch_size * seq_len, input_dim)
        tokens_per_expert = tokens_per_expert.view(-1, topk) # 
        flatten_tokens_per_expert = tokens_per_expert.view(-1)
        
        sorted_tokens_per_expert = torch.argsort(flatten_tokens_per_expert, stable=True)
        permuted_tokens = hidden_states.index_select(0, sorted_tokens_per_expert // topk)

        num_local_tokens_per_expert = torch.histc(tokens_per_expert, bins=self.num_experts, min=0, max=self.num_experts)

        num_local_tokens_per_expert = num_local_tokens_per_expert.to(torch.device('cpu')).to(torch.int64)
        print(num_local_tokens_per_expert)

        fc1_output = gmm(permuted_tokens.to(torch.bfloat16), self.w1.to(torch.bfloat16), num_local_tokens_per_expert, trans_b=False)

        intermediate_states = self.activation_fn(fc1_output)
        fc2_output = gmm(intermediate_states, self.w2, num_local_tokens_per_expert, trans_b=False)

        return fc2_output

if __name__ == "__main__":
    # 定义参数
    num_experts = 4
    expert_intermediate_size = 64
    hidden_dim = 48
    output_dim = 48
    batch_size = 4
    seq_len = 6

    w1 = torch.randn(num_experts, hidden_dim, expert_intermediate_size * 2, dtype=torch.bfloat16).cuda()
    w2 = torch.randn(num_experts, expert_intermediate_size, hidden_dim, dtype=torch.bfloat16).cuda()

    # # 创建输入数据
    input_data = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32).cuda()  # 形状: (batch_size, seq_len, input_dim)
    # print(input_data.shape)

    # 创建专家索引（假设每个 token 随机分配2个专家）
    topk_nums = 2
    expert_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk_nums), dtype=torch.float32).cuda()  # 形状: (batch_size, seq_len, num_experts)
    # print(expert_indices)
    
    # 创建 Experts 实例
    # experts = Experts(num_experts, expert_intermediate_size, hidden_dim, output_dim, w1, w2) 
    # output = experts(input_data, expert_indices)
    # print(output.sum())  # 输出形状: (batch_size, seq_len, output_dim)

    # print(expert_indices)
    grouped_gemm_experts = GroupedGemmExperts(num_experts, expert_intermediate_size, hidden_dim, output_dim, w1, w2)
    output = grouped_gemm_experts(input_data, expert_indices)
    print(output)
