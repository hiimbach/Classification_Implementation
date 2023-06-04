import datetime
import torch 
import torchvision
import os 
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .data_loader import CustomDataset, data_split, write_file_classnames
from .metric import compare


class TrainingLoop():
    '''
    Create train task to train nn.Module model
    
    Arguments:
        model (torch.nn.Module): The classification model you want to train 
        data_path (str): The data path, contains folders represent for classes
        batch_size (int)
        loss_fn (torch.nn): Loss function/criterion
        optim_fn (torch.optim): Optimizer
        lr (float): Learning rate
        train_transform (torchvision.transforms): Transforms to augment and 
                                                    change training images to tensor
        val_transform (torchvision.transforms): Transforms to augment and 
                                                    change images to tensor
        (Optional)
        data_split_ratio (float): the ratio of training images / total images
    '''
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
        
        # Prepare data for training and evaluation
        train_data, val_data, self.class_names = data_split(data_path, split_ratio=data_split_ratio)
        self.num_class = len(self.class_names)
        train_dataset = CustomDataset(train_data, train_transform)
        val_dataset = CustomDataset(val_data, val_transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        self.val_total = len(val_dataset)
        
        # Define training device 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn.to(self.device)
        self.model.to(self.device)
        
        
    def train(self, n_epochs, save_name, eval_interval=5, pretrained_weight=None):
        '''
        Train model, save weights and logs
        
        Parameters:
            n_epochs (int): Number of epoch trained
            save_name (os.path or str): dir name to save weight checkpoint
            eval_interval (int): validate after a number of epochs
            pretrained_weight(os.path or str): path to pretrained weight
        '''
        
        # Load pretrained weight
        if pretrained_weight:
            self.model.load_state_dict(torch.load(pretrained_weight, map_location=self.device))
        
        # Prepare for saving 
        save_path = os.path.join('runs', f"{save_name}")
        
        # In case there is a folder named like save_path
        if os.path.exists(save_path):
            i = 1
            while os.path.exists(f"{save_path}_{i}"):
                i += 1
            save_path = f"{save_path}_{i}"
            
        # Create folder and save file class_names
        os.makedirs(save_path)
        write_file_classnames(class_names=self.class_names, save_name="class_names", save_path=save_path)
        
        # Save tensor log
        writer = SummaryWriter(save_path)
        
        # Create folder to save weights
        save_weight_path = os.path.join(save_path, "weights")
        os.mkdir(save_weight_path)
        
        # Max accuracy - used to find best checkpoint
        max_acc = 0
        
        print(f"{datetime.datetime.now()} Start train on device {self.device}")
        
        for epoch in range(1, n_epochs + 1):  
            # Training Phase
            self.model.train()
            print(f"Epoch {epoch}")
            train_losses = []
            
            # Batch training
            for images, labels in tqdm(self.train_loader, desc="Training"):
                # Write images to tensorboard
                img_grid = torchvision.utils.make_grid(images)
                writer.add_image('four_fashion_mnist_images', img_grid)
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Predict
                out = self.model(images)
                train_loss = self.loss_fn(out, labels)
                
                # Backpropagation
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.item())
                
                
            # End training phase
            mean_train_loss = sum(train_losses)/len(train_losses)
            print(f"{datetime.datetime.now()} Epoch {epoch}: Training loss: {mean_train_loss}")
            
            # Save last checkpoint and write result to tensorboard
            torch.save(self.model.state_dict(), os.path.join(save_weight_path, "last_ckpt.pt"))
            writer.add_scalar("Loss/train", mean_train_loss, epoch)
            
            ###########################################################################################
            # After (eval_interval) epoch, go validating 
            self.model.eval()
            if epoch == 1 or epoch % eval_interval == 0:
                with torch.no_grad():
                    val_losses = []
                    total_correct = 0
                    total = self.val_total
                    
                    for images, labels in tqdm(self.val_loader, desc="Validating"):
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        out = self.model(images)
                        val_loss = self.loss_fn(out, labels)
                        correct = compare(out, labels)  
                        
                        val_losses.append(val_loss.item())    
                        total_correct += correct
                    
                # Calculate loss and accuracy
                acc = total_correct / total
                mean_val_loss = sum(val_losses)/len(val_losses)
                
                # Replace best checkpoint if loss < min_loss:
                if acc > max_acc:
                    max_acc = acc
                    torch.save(self.model.state_dict(), os.path.join(save_weight_path, "best_ckpt.pt"))
            
                # Write to tensorboard
                writer.add_scalar("Loss/val", mean_val_loss, epoch)
                writer.add_scalar("Accuracy/val", acc, epoch)
                
                # End validating
                print(f"{datetime.datetime.now()} Val Loss {mean_val_loss}")
                print(total_correct, total)
                print(f"{datetime.datetime.now()} Val Accuracy {acc}")
                print("="*70)
                print("")
                
        writer.close()
        return
        
