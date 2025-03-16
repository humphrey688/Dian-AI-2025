import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, latent_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.d_head = latent_dim // num_heads

        # 潜在投影矩阵
        self.q_proj = nn.Linear(d_model, latent_dim)
        self.kv_proj = nn.Linear(d_model, 2*latent_dim)
        
        # 输出变换
        self.out_proj = nn.Linear(latent_dim, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 潜在空间投影
        q = self.q_proj(query)  # [B, Lq, l]
        k, v = torch.chunk(self.kv_proj(key), 2, dim=-1)  # [B, Lk, l]
        
        # 多头拆分
        q = q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # 合并输出
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        return self.out_proj(context)

# 测试用例
def test_mla():
    d_model = 512
    mla = MultiHeadLatentAttention(latent_dim=64)
    x = torch.randn(2, 10, d_model)  # batch=2, seq=10
    output = mla(x, x, x)
    assert output.shape == (2, 10, d_model), "✅ MLA形状验证通过！"
    print("✅ MLA测试通过")

test_mla()