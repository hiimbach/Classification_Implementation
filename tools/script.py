import os
import sys
import torch
import shutil

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from model.resnet import ResNet50
from model.vit import ViT
from torchvision.models import resnet50
from torch import nn
from torchsummary import summary

def script_model(name, file_names_path, weight_path=None, save_path='model/scripted_model'):
    model = ResNet50(9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load weight
    if weight_path: 
        # Load trained weight
        model.load_state_dict(torch.load(weight_path, map_location=device))
        summary(model, (3, 224, 224))
        
    save_path = os.path.join(save_path, name)        
    os.mkdir(save_path)
        
    # Script model
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, os.path.join(save_path, "scripted_model.pt"))

    # Copy file name 
    shutil.copy(file_names_path, os.path.join(save_path, "class_names.txt"))
    
if __name__ == '__main__':
    img_size = 224
    weight_path = 'runs/colab/weights/last_ckpt3.pt'
    save_path = 'model/scripted_model'
    file_names_path = 'runs/colab/class_names.txt'
    script_model('resnet50',file_names_path=file_names_path, weight_path=weight_path, save_path=save_path)