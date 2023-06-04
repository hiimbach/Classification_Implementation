import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from torch import nn, optim
from torchvision import transforms

from model.resnet import ResNet50
from utils.train_loop import TrainingLoop
 
 
# Define model and params
model = ResNet50(9)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])  

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = "data/mushrooms"
batch_size = 4
loss_fn = nn.CrossEntropyLoss()  
optim_fn = optim.Adam

# Create train task
train_task = TrainingLoop(model, data_path, batch_size, loss_fn, optim_fn, 0.001, train_transform, val_transform)

# Start train
train_task.train(n_epochs=10,save_name="test", eval_interval=1)
