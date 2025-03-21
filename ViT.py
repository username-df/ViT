import os
import torch
from torch import nn
from PatchEmbed import PatchEmbed
from Encoder import Encoder

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_blocks, num_classes):
        super().__init__()
        num_patches = (img_size ** 2) // (patch_size ** 2)
        self.embed = PatchEmbed(patch_size, num_patches, embed_dim)

        self.encoder = nn.Sequential(*[
            Encoder(embed_dim, num_heads) for _ in range(num_blocks)
        ])

        self.final_MLP =  nn.Linear(embed_dim, num_classes)

    def forward(self, image):
        x = self.embed(image)
        x = self.encoder(x)

        # use the CLS token as input to the classifier
        x = self.final_MLP(x[:, 0, :])
        return x
    
    def save(self, file_name):
        model_folder_path = './saved_models'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        
        torch.save({
            'model_state_dict': self.state_dict(), 
            }, file_name)

    def load(self, file_name):
        model_folder_path = './saved_models'
        file_path = os.path.join(model_folder_path, file_name)

        if os.path.exists(file_path):
            load_model = torch.load(file_path, map_location=torch.device('cpu'))

            self.load_state_dict(load_model['model_state_dict'])
            
        else:
            print("No saved model found") 