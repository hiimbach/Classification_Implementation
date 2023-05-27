import os 
import torch
# import ipdb 
import random

from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

def custom_dataloader(dir, batch_size, split_ratio=0.8):
    '''
    Parameters:
        dir: data directory, format as:
            0/a.jpg
            1/b.jpg
            ...
        batch_size (int)
        split_ratio: the ratio of train_data/all_data
    
    Return:
        train_loader (dict): {'img_path': list(str), 'label': list(Tensor (batch_size))}
        val_loader (dict): {'img_path': list(str), 'label': list(Tensor (batch_size))}
    '''
    
    # This variable is used to store all data read
    data = {}
    num_classes = 0
    class_names = []
    class_idx = -1
    
    # First we read all data from the dir
    # Each folder represents for a class
    for folder in os.listdir(dir):
        num_classes += 1
        folder_path = os.path.join(dir, folder)
        class_names.append(folder)
        class_idx += 1
        
        data[folder] = {'img_path': [], 'label': []}
        data_folder = data[folder]
        
        for item in os.listdir(folder_path):
            img_path = os.path.join(folder_path, item)

            # In each folder in the data dictionary, we append img and corresponding label in 2 list
            data_folder['img_path'].append(img_path)
            data_folder['label'].append(class_idx)
            
    # Now we create train dict and val dict to split data
    train_data = {'img_path': [], 'label': []}
    val_data = {'img_path': [], 'label': []}

    # Split data to train and val
    for folder in data:
        data_folder = data[folder]
        
        # Choose img and corresponding label randomly
        idx = range(len(data_folder['label']))
        print(len(data_folder['label']))
        train_idx, test_idx = train_test_split(idx, test_size=(1-split_ratio))
        for i in train_idx:
            train_data['img_path'].append(data_folder['img_path'][i])
            train_data['label'].append(data_folder['label'][i])
            
        for i in test_idx:
            val_data['img_path'].append(data_folder['img_path'][i])
            val_data['label'].append(data_folder['label'][i])
            
    
    return create_batch(train_data, batch_size, num_classes), create_batch(val_data, batch_size, num_classes), class_names
    
    
def create_batch(data, batch_size, num_classes):
    # Use to create batch and generate dataloader
    '''
    Argument: 
        data (dict): {'img_path': [], 'label': []}
        
    Return:
        data_loader: {'img_path': list(str), 'label': list(Tensor (batch_size))}

    '''
    data_loader = {'img_path': [], 'label': [], 'num_classes': num_classes}
    data_len = len(data['label'])
    
    # Shuffle the data for better batch
    random.shuffle(data['img_path'])
    random.shuffle(data['label'])
    
    i = 0
    while (i+batch_size) < data_len:
        img_batch = data['img_path'][i:i+batch_size]
        label_batch = torch.tensor(data['label'][i:i+batch_size])
        
        data_loader['img_path'].append(img_batch)
        data_loader['label'].append(label_batch)
        
        i += batch_size
        
    # Because there is some little data left which is not enough to create a full batch,
    # we group it to the final batch 
    img_batch = data['img_path'][i:]
    label_batch = torch.tensor(data['label'][i:])
    
    data_loader['img_path'].append(img_batch)
    data_loader['label'].append(label_batch)
    
    return data_loader


def filename_to_tensor(files, transform):
    '''
    Parameters:
        files (list): list of file name (str)
        transform (torchvision.transforms.transforms.Compose): requires to transform to Tensor after augment
        
    Return:
        img_batch (torch.Tensor)    
    '''
    
    # This list is used to store all tensor and concat at the end
    tensor_list = []
    
    # Read all files and transform it to tensor
    for img_path in files:
        img = Image.open(img_path)
        # Unsqueeze is used to expand dim to concat later
        tensor_list.append(torch.unsqueeze(transform(img), dim=0))
        
    return torch.cat(tensor_list)
    
# tf = transforms.Compose([
#     transforms.Resize([640,640]),
#     transforms.ToTensor()
# ])

# a = custom_dataloader("test_ds", 8)
# ipdb.set_trace()