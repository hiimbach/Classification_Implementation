import os
import sys
import torch

from torchvision import transforms
 
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.vit import ViT

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tf = transforms.Compose([
    transforms.Resize([640,640]),
    transforms.ToTensor()
])

weight_path = 'weights/50_ckpt.pt'
model.load_state_dict(torch.load(weight_path, map_location=device))