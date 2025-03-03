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