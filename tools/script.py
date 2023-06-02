import os
import sys
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from model.resnet import ResNet50
from model.vit import ViT
from torchvision.models import resnet50
from torch import nn
from torchsummary import summary

img_size = 224
weight_path = 'weights/model1.pth'
save_path = ''

def script_model(name, weight_path=None, save_path='model/scripted_model'):
    model = ResNet50(9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load weight
    if weight_path: 
        # Load trained weight
        model.load_state_dict(torch.load(weight_path, map_location=device))
        summary(model, (3, 224, 224))
        
    # Script model
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, os.path.join(save_path, name))

if __name__ == '__main__':
    script_model('resnet50.pt', weight_path='weights/')