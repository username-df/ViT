import torch
from torch import nn

# embed image patches
class PatchEmbed(nn.Module):
    def __init__(self, num_patches, patch_size, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.lin_proj = nn.Linear(3*patch_size**2, embed_dim) # linear projection of patches
        self.pos_enc = nn.Parameter(torch.randn(num_patches+1, embed_dim), requires_grad=True) # learnable positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True) # learnable CLS token

    def forward(self, patches):
        # patches = (batch_size, num_patches, 3*patch_size**2)
        proj = self.lin_proj(patches) # (batch_size, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(proj.shape[0], -1, -1) # account for batch size -> (batch_size, 1, embed_dim)
        embeddings = torch.cat((cls_tokens, proj), axis=1) # (batch_size, num_patches + 1, embed_dim)
        embeddings += self.pos_enc # add positional encoding
        return embeddings