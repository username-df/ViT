import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, qkv_output):
        super().__init__()
        self.qkv_output = qkv_output
        self.scale = self.qkv_output ** -0.5

        # (batch, num_patches, qkv_output)
        self.query = nn.Linear(embed_dim, self.qkv_output)
        self.key = nn.Linear(embed_dim, self.qkv_output)
        self.value = nn.Linear(embed_dim, self.qkv_output)

    def forward(self, Q, K, V):
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        # compute softmax(QK^T / sqrt(dk))V
        # using transpose since .T affects batch dimension
        scaled = (Q @ K.transpose(-2, -1)) * self.scale
        softmax = nn.functional.softmax(scaled, dim=-1)
        return softmax @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert (embed_dim % num_heads == 0), "Embed dimension can not be divided by number of heads."
        self.head_size = embed_dim // num_heads
        self.HeadsList = nn.ModuleList([SelfAttention(embed_dim, self.head_size) for _ in range(num_heads)])
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attention = torch.cat([h(x, x, x) for h in self.HeadsList], dim=-1)
        attention = self.output_proj(attention)
        attention = self.dropout(attention)
        return attention
