import os
import sys
import torch

from PIL import Image
from torchvision import transforms
 
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.vit import ViT
from utils.data_loader import filename_to_tensor

img_size = 640
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tf = transforms.Compose([
    transforms.Resize([640,640]),
    transforms.ToTensor()
])

# weight_path = 'weights/best_ckpt.pt'
# model.load_state_dict(torch.load(weight_path, map_location=device))

path = "mushrooms_test/Agaricus"

# Infer
def inference(model, path, transform, weight_path=None, batch_size=None):
    # Check path is dir or file:
    if not os.path.exists(path):
        print("No path")
    if os.path.isfile(path):
        img = Image.open(path)
        data_loader = [[img]]
        print("Is file")
    elif os.path.isdir(path):
        if not batch_size:
            print("If path is directory, please choose batch size")
            return
        else:
            img_list = [os.path.join(path, item) for item in os.listdir(path)]
            data_len = len(img_list)
            data_loader = []
            i = 0
            # Load into batches
            while (i+batch_size) < data_len:
                path_batch = img_list[i:i+batch_size]
                data_loader.append(path_batch)
                i += batch_size
            print("Is dir")
    else:
        print("Wtf")
        return
    
    # Infer
    result = {}
    with torch.no_grad():
        for path_batch in data_loader:
            print(path_batch)
            img_batch = filename_to_tensor(path_batch, transform)
            
            # Use ViT for prediction
            outputs = model(img_batch)
            _, predicted = torch.max(outputs, dim=1)
            assert(len(predicted) == len(path_batch))
            
            for k in range(len(path_batch)):
                result[path_batch[k]] = predicted[k]
    
    return result

print(inference(model, path, tf, batch_size=8))