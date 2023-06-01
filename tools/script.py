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

img_size = 224
weight_path = 'weights/model1.pth'
save_path = ''

def script_model(name, img_size, weight_path=None, save_path='model/scripted_model'):
    # model = ViT(
    #     image_size = img_size,
    #     patch_size = 32,
    #     num_classes = 9,
    #     dim = 1024,
    #     depth = 6,
    #     heads = 16,
    #     mlp_dim = 2048,
    #     dropout = 0.1,
    #     emb_dropout = 0.1
    # )
    # model = ResNet50(9)
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 9)

    if weight_path: 
        # Load trained weight
        model.load_state_dict(torch.load(weight_path))
    
    # Script model
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, os.path.join(save_path, name))

if __name__ == '__main__':
    script_model('wtf.pt', 224)