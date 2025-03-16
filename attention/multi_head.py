import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 初始化线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询向量变换
        self.W_k = nn.Linear(d_model, d_model)  # 键向量变换
        self.W_v = nn.Linear(d_model, d_model)  # 值向量变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

        # 保证维度可被头数整除
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

    def forward(self, Q, K, V, mask=None):
        """
        前向传播过程
        Args:
            Q: 查询向量 [batch_size, seq_len, d_model]
            K: 键向量 [batch_size, seq_len, d_model]
            V: 值向量 [batch_size, seq_len, d_model]
            mask: 掩码矩阵 [batch_size, seq_len, seq_len]
        Returns:
            output: 输出向量 [batch_size, seq_len, d_model]
            attn_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = Q.size(0)
        
        # 1. 线性变换并分头
        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, L, d_k]
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, L, d_k]
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, L, d_k]

        # 2. 计算缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))  # [B, h, L, L]
        
        # 3. 应用mask（如果有）
        if mask is not None:
            # mask形状: [B, 1, 1, L]（padding mask）或 [B, 1, L, L]（sequence mask）
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 4. 计算注意力权重和上下文向量
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, h, L, L]
        context = torch.matmul(attn_weights, v)  # [B, h, L, d_k]
        
        # 5. 合并多头结果
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # [B, L, d_model]
        
        # 6. 输出变换
        output = self.W_o(context)  # [B, L, d_model]
        
        return output, attn_weights


# 测试用例（验证所有核心功能）
def test_multi_head():
    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    # 初始化模块
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 测试数据
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 基础形状测试
    output, attn_weights = mha(Q, K, V)
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误，实际得到 {output.shape}"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), f"注意力权重形状错误，实际得到 {attn_weights.shape}"
    
    # 2. Mask功能测试
    mask = torch.tril(torch.ones(batch_size, seq_len, seq_len))  # 下三角mask
    masked_output, masked_attn = mha(Q, K, V, mask=mask.unsqueeze(1))  # 增加头维度
    assert not torch.allclose(attn_weights, masked_attn), "Mask功能失效"
    
    # 3. 梯度回传测试
    output.sum().backward()
    for param in mha.parameters():
        assert param.grad is not None, "梯度未正确计算"
    
    # 4. 数值范围验证
    assert torch.all(attn_weights >= 0) and torch.all(attn_weights <= 1), "注意力权重超出[0,1]范围"
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1))), "注意力权重未归一化"
    
    print("所有测试通过：形状、mask、梯度、数值范围验证成功！")

# 执行测试
test_multi_head()