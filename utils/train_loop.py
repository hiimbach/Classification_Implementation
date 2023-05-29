import datetime
import torch 
import os 
from tqdm import tqdm


from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter

from .data_loader import filename_to_tensor 
from .accuracy import accuracy

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, transform, saved_path, eval_interval=5):
    '''
    Parameters:
        n_epochs (int): Number of epoch trained
        optimizer (torch.optim): Optimizer
        model (nn.Module): training model
        loss_fn (torch.nn.modules.loss): Loss function
        train_loader (dict): Custom training data loader in utils/dataloader.py {'img_path': [...], 'label': [...]}
        val_loader (dict): Custom validate data loader in utils/dataloader.py {'img_path': [...], 'label': [...]}
        transform (torchvision.transforms.transforms): requires to transform to Tensor after augment
        saved_path (os.path or str): dir to save weight checkpoint
        
        eval_interval (int): validate after a number of epochs
    '''
    
    # Prepare for tensorboard
    writer = SummaryWriter("runs/mushrooms")
    
    # Max accuracy - used to find best checkpoint
    max_acc = 0
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"{datetime.datetime.now()} Start train on device {device}")
    
    
    for epoch in range(1, n_epochs + 1):  
        print(f"Epoch {epoch}")
        # Parameters: Loss train, loss val and accuracy (correct/total)
        loss_train = 0.0
        loss_val = 0.0
        correct = 0
        total = 0
        
        # Batch training
        for i in tqdm(range(len(train_loader['label'])), desc="Training"):
            # Read data from data loader and transform filename into tensor
            filenames = train_loader['img_path'][i]
            labels = train_loader['label'][i]
            img_batch = filename_to_tensor(filenames, transform).to(device)
            
            # Use ViT for prediction
            outputs = model(img_batch)
            
            # Convert label to one-hot format to calculate loss
            num_classes= train_loader['num_classes']
            actual = one_hot(labels, num_classes).type(torch.float32).to(device)
            loss = loss_fn(outputs, actual)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Sum all loss to calculate the loss of the epoch
            loss_train += loss.item()
        
        # Average train loss over batches
        avg_loss = loss_train / len(train_loader['label'])
        writer.add_scalar("Loss/train", avg_loss, epoch)
        
        # Print loss of epoch
        print(f"{datetime.datetime.now()} Train Loss {avg_loss}")
        # for name, param in :
        # print(model.state_dict().items()[0], model.state_dict().items()[1])
        
        # Save last checkpoint
        torch.save(model.state_dict(), os.path.join(saved_path, "last_ckpt.pt"))

        # Eval interval
        # After a number of epoch, evaluate
        if epoch == 1 or epoch % eval_interval == 0:
            print("-"*50)   
            with torch.no_grad():
                for k in tqdm(range(len(val_loader['label'])), desc="Validate"):
                    # Read data from data loader and transform filename into tensor
                    val_filenames = val_loader['img_path'][k]
                    val_labels = val_loader['label'][k].to(device)
                    val_img_batch = filename_to_tensor(val_filenames, transform).to(device)
                    
                    # Use ViT for prediction, and call torch.max for final pred
                    val_outputs = model(val_img_batch)
                    batch_loss_val = loss_fn(val_outputs, val_labels)
                    _, predicted = torch.max(val_outputs, dim=1)
                    
                    # print(val_outputs[0])
                    # print(predicted[0], val_labels[0])
                    
                    # Sum all loss to calculate the loss of the epoch
                    loss_val += batch_loss_val.item()

                    # Sum total and correct
                    total += val_labels.shape[0] 
                    correct += int((predicted == val_labels).sum()) 
                acc = correct / total
                avg_val_loss = loss_val / len(val_loader)
                # Replace best checkpoint if loss < min_loss:
                if acc > max_acc:
                    max_acc = acc
                    torch.save(model.state_dict(), os.path.join(saved_path, "best_ckpt.pt"))
                    
            # Write to tensorboard
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val", acc, epoch)
                    
            
            # Print validation loss
            print(f"{datetime.datetime.now()} Val Loss {avg_val_loss}")
            print(correct, total)
            print(f"{datetime.datetime.now()} Val Accuracy {acc}")
            print("="*70)
            print("")
            
    writer.close()
    return
        
            
        
    
        