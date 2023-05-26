import torch
import torchvision.transforms as transforms
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os 

from model.vit import ViT
from utils.train_loop import training_loop

# Init model
model = ViT(
    image_size = 640,
    patch_size = 16,
    num_classes = 4,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# Define optimizer
optimizer = optim.Adam()

# Define transform to augment and create dataloader 
transform = transforms.Compose([
                transforms.Resize([640,640]), # Resizing the image as the VGG only take 640 x 640 as input size
                transforms.RandomVerticalFlip(), # Flip the data vertically
                transforms.ToTensor(),
            ])

