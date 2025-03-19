import torch
from torch import nn
from Attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.MSA = MultiHeadAttention(embed_dim, num_heads)

        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

        self.LN1 = nn.LayerNorm(embed_dim)
        self.LN2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x + self.MSA(self.LN1(x))
        x = x + self.dropout(self.MLP(self.LN2(x)))
        return x