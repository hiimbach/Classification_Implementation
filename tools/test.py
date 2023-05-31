import torch
import os
import sys
import ipdb

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from torch import nn, optim
from torchvision import models, transforms


from model.vit import  ViT
from utils.train_loop import TrainingLoop
from utils.data_loader import custom_dataloader, filename_to_tensor
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights
 
 
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
newfc = torch.nn.Linear(in_features=model.fc.in_features, out_features=9, bias=True)
model.fc = newfc


tf = transforms.Compose([
    transforms.Resize([640,640]),
    transforms.ToTensor()
])

data_path = "data/mushrooms_test"
batch_size = 4
loss_fn = nn.CrossEntropyLoss()  
optim_fn = optim.Adam

train_task = TrainingLoop(model, data_path, batch_size, loss_fn, optim_fn, 0.001, tf)

train_task.fit(10, "test", 1)