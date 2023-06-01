import os
import sys
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from model.resnet import ResNet50
from model.vit import ViT

img_size = 224
weight_path = 'weights/best_ckpt.pt'
save_path = ''

def script_model(name, img_size, weight_path=None, save_path='model/scripted_model'):
    model = ViT(
        image_size = img_size,
        patch_size = 32,
        num_classes = 9,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    # model = ResNet50(9)

    if weight_path: 
        # Load trained weight
        model.load_state_dict(torch.load(weight_path))
    
    # Script model
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, os.path.join(save_path, name))

if __name__ == '__main__':
    script_model('test.pt', 224)