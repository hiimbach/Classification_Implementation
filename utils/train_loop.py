import datetime
import torch 
import torchvision
import os 
from tqdm import tqdm

from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from .data_loader import filename_to_tensor, create_data_loader, data_split
from .metric import accuracy, compare


class TrainingLoop():
    def __init__(self, model: torch.nn.Module, 
                        data_path: str, 
                        batch_size: int, 
                        loss_fn, 
                        optim_fn: torch.optim, 
                        lr: float, 
                        train_transform: torchvision.transforms, 
                        val_transform: torchvision.transforms, 
                        data_split_ratio = 0.8):
        
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optim_fn(self.model.parameters(), lr)
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        # Specfically for loading data and evaluation
        self.train_data, self.val_data, self.num_classes = data_split(data_path, split_ratio=data_split_ratio)
        self.num_train_batches = len(self.train_data['label'])//batch_size + 1  # Number of batches of train loader
        self.len_val_data = len(self.val_data['label'])
        self.num_val_batches = self.len_val_data//batch_size + 1  # Number of batches of val loader
        
        # Define training device 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
        
    def training_step(self, img_path_batch, labels):
        # Create tensor batches from img paths
        img_batch = filename_to_tensor(img_path_batch, self.train_transform).to(self.device)
        
        # Predict
        out = self.model(img_batch)
        actual = self.one_hot_label(labels)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.loss_fn(out, actual)
        loss.backward()
        self.optimizer.step()
                
        return loss.item()

        
    def validation_step(self, img_path_batch, labels):
        with torch.no_grad():
            # Create tensor batches from img paths
            img_batch = filename_to_tensor(img_path_batch, self.val_transform).to(self.device)
            
            # Calculate loss and accuracy
            out = self.model(img_batch)
            actual = self.one_hot_label(labels)
            loss = self.loss_fn(out, actual)
            correct = compare(out, labels)  
            
        return {'val_loss': loss.detach(), 'correct': correct}
        
        
    def one_hot_label(self, labels):
        return one_hot(labels, self.num_classes).type(torch.float32).to(self.device)
            
        
    def train(self, n_epochs, save_name, eval_interval=5):
        '''
        Parameters:
            n_epochs (int): Number of epoch trained
            save_weight (os.path or str): dir name to save weight checkpoint
            
            eval_interval (int): validate after a number of epochs
        '''
        # Prepare for saving and tensorboard
        save_path = os.path.join('runs', f"{save_name}{datetime.datetime.now()}" )
        if os.path.exists(save_path):
            print(f"There is a folder named {save_name} in runs/")
            return
        else:
            os.makedirs(save_path)
        writer = SummaryWriter(f"runs/{save_name}")
        
        # Max accuracy - used to find best checkpoint
        max_acc = 0
        
        print(f"{datetime.datetime.now()} Start train on device {self.device}")
        
        for epoch in range(1, n_epochs + 1):  
            # Training Phase
            print(f"Epoch {epoch}")
            
            # Create data_loader
            train_loader = create_data_loader(self.train_data, self.batch_size, self.num_classes)
            train_losses = []
                        
            # Batch training
            for i in tqdm(range(len(train_loader['label'])), desc="Training"):
                # Read data from train_loader
                img_path_batch = train_loader['img_path'][i]
                labels = train_loader['label'][i].to(self.device)
                
                # Training step get loss
                train_loss = self.training_step(img_path_batch, labels)
                train_losses.append(train_loss)
                
            # End training phase
            mean_train_loss = sum(train_losses)/len(train_losses)
            print(f"{datetime.datetime.now()} Epoch {epoch}: Training loss: {mean_train_loss}")
            
            # Save last checkpoint and write result to tensorboard
            torch.save(self.model.state_dict(), os.path.join(save_path, "last_ckpt.pt"))
            writer.add_scalar("Loss/train", mean_train_loss, epoch)
            
            ###########################################################################################
            # After (eval_interval) epoch, go validating 
            if epoch == 1 or epoch % eval_interval == 0:
                # Creat data_loader
                val_loader = create_data_loader(self.val_data, self.batch_size, self.num_classes)
                val_losses = []
                correct = 0
                total = self.len_val_data
                
                for k in tqdm(range(len(val_loader['label'])), desc="Validate"):
                    # Read data from train_loader
                    img_path_batch = val_loader['img_path'][k]
                    labels = val_loader['label'][k].to(self.device)
                    
                    # Training step get loss
                    result = self.validation_step(img_path_batch, labels)
                    val_loss = result['val_loss']
                    val_losses.append(val_loss)    
                    correct += result['correct']
                    
                acc = correct / total
                mean_val_loss = sum(val_losses)/len(val_losses)
                
                # Replace best checkpoint if loss < min_loss:
                if acc > max_acc:
                    max_acc = acc
                    torch.save(self.model.state_dict(), os.path.join(save_path, "best_ckpt.pt"))
            
                # Write to tensorboard
                writer.add_scalar("Loss/val", mean_val_loss, epoch)
                writer.add_scalar("Accuracy/val", acc, epoch)
                
                # End validating
                print(f"{datetime.datetime.now()} Val Loss {mean_val_loss}")
                print(correct, total)
                print(f"{datetime.datetime.now()} Val Accuracy {acc}")
                print("="*70)
                print("")
                
        writer.close()
        return
        
        
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, saved_path, eval_interval=5):
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
        model.train()
        for i in tqdm(range(len(train_loader['label'])), desc="Training"):
            # Read data from data loader and transform filename into tensor
            filenames = train_loader['img_path'][i]
            labels = train_loader['label'][i]
            img_batch = filename_to_tensor(filenames, transform).to(device)
            
            # Convert label to one-hot format to calculate loss
            num_classes= train_loader['num_classes']
            actual = one_hot(labels, num_classes).type(torch.float32).to(device)
            loss = loss_fn(outputs, actual)
            
            # Use ViT for prediction
            outputs = model(img_batch)
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
                avg_val_loss = loss_val / len(val_loader['label'])
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
        
            
        
    
        