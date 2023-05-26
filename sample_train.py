import torch
import numpy as np
import torch.nn.functional as F

from torch import nn, optim
from torchvision import transforms
from .utils.data_loader import custom_dataloader, filename_to_tensor
from .utils.train_loop import training_loop

from model.vit import  ViT
model = ViT(
    image_size = 640,
    patch_size = 32,
    num_classes = 4,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

tf = transforms.Compose([
    transforms.Resize([640,640]),
    transforms.ToTensor()
])

train_loader, val_loader = custom_dataloader("vit_ds_class", 8)

sample = train_loader['img_path'][0]
img_batch = filename_to_tensor(sample, tf)
preds = model(img_batch)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()  

n_epochs = 50
transform = tf
saved_path = 'weights'

training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, transform, saved_path, eval_interval=3)
