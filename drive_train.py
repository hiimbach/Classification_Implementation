# import torch
# import numpy as np
import torch.nn.functional as F

from torch import nn, optim
from torchvision import transforms

from utils.data_loader import custom_dataloader
from utils.train_loop import training_loop
from model.vit import  ViT
from model.resnet import ResNet50

train_loader, val_loader, classes_names = custom_dataloader("/content/drive/MyDrive/mushrooms", 32)
num_classes = len(classes_names)
img_size = 640

tf = transforms.Compose([
    transforms.Resize([img_size,img_size]),
    transforms.ToTensor()
])

# model = ViT(
#     image_size = img_size,
#     patch_size = 32,
#     num_classes = num_classes,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

model = ResNet50(num_classes)

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()  

n_epochs = 50
transform = tf
saved_path = ''

training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, transform, saved_path, eval_interval=1)
