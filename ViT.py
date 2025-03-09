from torch import nn
from PatchEmbed import PatchEmbed
from Encoder import Encoder

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_classes):
        super().__init__()
        num_patches = (img_size ** 2) // (patch_size ** 2)
        self.embed = PatchEmbed(patch_size, num_patches, embed_dim)
        self.encoder = Encoder(embed_dim, num_heads)
        self.final_MLP = nn.Linear(embed_dim, num_classes)

    def forward(self, image):
        x = self.embed(image)
        x = self.encoder(x)

        # use the CLS token as input to the classifier
        x = self.final_MLP(x[:, 0, :])
        return x