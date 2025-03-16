# File: GQA_MQA.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- 多查询注意力 (MQA) -------------------------
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 参数定义
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_kv = nn.Linear(d_model, 2 * self.d_k)  # 共享键值变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.size()
        
        # 关键修复点1：正确分头查询
        q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, L, d_k]
        
        # 关键修复点2：正确处理共享键值
        kv = self.W_kv(K).view(batch_size, -1, 2, self.d_k)  # [B, L, 2, d_k]
        k, v = kv.chunk(2, dim=2)  # [B, L, 1, d_k]
        k = k.permute(0, 2, 1, 3)  # [B, 1, L, d_k]
        v = v.permute(0, 2, 1, 3)  # [B, 1, L, d_k]

        # 注意力计算（维度对齐）
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # [B, h, L, d_k]
        
        # 合并多头
        output = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.W_o(output)

# ------------------------- 分组查询注意力 (GQA) -------------------------
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=512, num_groups=4):
        super().__init__()
        assert d_model % num_groups == 0, "d_model必须能被num_groups整除"
        self.num_groups = num_groups
        self.d_k = d_model // num_groups
        
        # 参数定义
        self.W_q = nn.Linear(d_model, d_model)
        self.W_kv = nn.ModuleList([
            nn.Linear(d_model, 2 * self.d_k) for _ in range(num_groups)
        ])
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.size()
        
        # 查询分头处理 
        q = self.W_q(Q).view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)  # [B, g, L, d_k]
        
        # 分组处理
        group_outputs = []
        for group_idx in range(self.num_groups):
            # 当前组的键值变换
            kv = self.W_kv[group_idx](K).view(batch_size, seq_len, 2, self.d_k)
            k, v = kv.chunk(2, dim=2)          # k/v形状: [B, L, 1, d_k]
            k = k.squeeze(2)                    # 移除冗余维度 → [B, L, d_k]
            v = v.squeeze(2)                    # 移除冗余维度 → [B, L, d_k]
            
            # 注意力计算
            q_group = q[:, group_idx]           # [B, L, d_k]
            attn_scores = torch.matmul(q_group, k.transpose(-2, -1))  # [B, L, L]
            attn_scores = attn_scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
                
            attn_weights = F.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, v)  # [B, L, d_k]
            group_outputs.append(context)
        
        # 合并所有组 
        combined = torch.cat(group_outputs, dim=-1)  # [B, L, g*d_k]
        return self.W_o(combined)  # [B, L, d_model]
# ------------------------- 测试函数 -------------------------
def test_gqa_mqa():
    d_model = 512
    batch_size = 2
    seq_len = 10
    
    # 测试数据（确保Q/K/V长度一致）
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    
    # MQA测试
    mqa = MultiQueryAttention(d_model, num_heads=8)
    out_mqa = mqa(input_tensor, input_tensor, input_tensor)
    assert out_mqa.shape == (batch_size, seq_len, d_model), f"MQA形状错误：{out_mqa.shape}"
    
    # GQA测试
    gqa = GroupedQueryAttention(d_model, num_groups=4)
    out_gqa = gqa(input_tensor, input_tensor, input_tensor)
    assert out_gqa.shape == (batch_size, seq_len, d_model), f"GQA形状错误：{out_gqa.shape}"
    
    print("所有测试通过！")

if __name__ == "__main__":
    test_gqa_mqa()

