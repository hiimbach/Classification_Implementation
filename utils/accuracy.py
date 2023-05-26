import torch 
from .data_loader import filename_to_tensor

def accuracy(model, dataloader, transform):
    correct = 0
    total = 0

    # # torch.no_grad() not to let model update weights
    # with torch.no_grad():  
    #     for imgs, labels in loader: 
    #         outputs = model(imgs)
    #         _, predicted = torch.max(outputs, dim=1) 
    #         total += labels.shape[0] 
    #         correct += int((predicted == labels).sum())   
    
         
    with torch.no_grad():
        for k in range(len(dataloader['label'])):
            # Read data from data loader and transform filename into tensor
            filenames = dataloader['img_path'][k]
            labels = dataloader['label'][k]
            img_batch = filename_to_tensor(filenames, transform)
            
            # Use ViT for prediction, then calculate the loss 
            outputs = model(img_batch)
            _, predicted = torch.max(outputs, dim=1)
            
            # Sum total and correct
            total += labels.shape[0] 
            correct += int((predicted == labels).sum())  

    print(f"Accuracy: {correct / total}")