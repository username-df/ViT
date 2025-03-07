import torch
from torch import nn
from Attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.MSA = MultiHeadAttention(embed_dim, num_heads=4)

        self.MLP = nn.Sequential([
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        ])

        self.LN1 = nn.LayerNorm(embed_dim)
        self.LN2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.MSA(self.LN1(x))
        x = x + self.MLP(self.LN2(x))
        return x