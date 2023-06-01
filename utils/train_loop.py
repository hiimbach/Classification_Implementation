import datetime
import torch 
import torchvision
import os 
from tqdm import tqdm

from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .data_loader import filename_to_tensor, data_split, CustomDataset
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
        
        # Prepare data for training and evaluation
        train_data, val_data, self.num_classes = data_split(data_path, split_ratio=data_split_ratio)
        train_dataset = CustomDataset(train_data, train_transform)
        val_dataset = CustomDataset(val_data, val_transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        self.val_total = len(val_dataset)
        
        # Define training device 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn.to(self.device)
        self.model.to(self.device)
        
    # def training_step(self, images, labels):
    #     # Predict
    #     out = self.model(images)
    #     # actual = self.one_hot_label(labels)
        
    #     # Backpropagation
    #     self.optimizer.zero_grad()
    #     loss = self.loss_fn(out, labels)
    #     loss.backward()
    #     self.optimizer.step()
                
    #     return loss.item()

        
    # def validation_step(self, images, labels):
    #     with torch.no_grad():
    #         # Calculate loss and accuracy
    #         out = self.model(images)
    #         # actual = self.one_hot_label(labels)
    #         loss = self.loss_fn(out, labels)
    #         correct = compare(out, labels)  
            
    #     return {'val_loss': loss.detach(), 'correct': correct}
        
        
    # def one_hot_label(self, labels):
    #     return one_hot(labels, self.num_classes).type(torch.float32).to(self.device)
            
        
    def train(self, n_epochs, save_name, eval_interval=5):
        '''
        Parameters:
            n_epochs (int): Number of epoch trained
            save_name (os.path or str): dir name to save weight checkpoint
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
            self.model.train()
            print(f"Epoch {epoch}")
            train_losses = []
            
            # Batch training
            for images, labels in tqdm(self.train_loader, desc="Training"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Predict
                out = self.model(images)
                train_loss = self.loss_fn(out, labels)
                print(train_loss)
                
                # Backpropagation
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.item())
                
                print(self.model.fc.weight)
                
            # End training phase
            mean_train_loss = sum(train_losses)/len(train_losses)
            print(f"{datetime.datetime.now()} Epoch {epoch}: Training loss: {mean_train_loss}")
            
            # Save last checkpoint and write result to tensorboard
            torch.save(self.model.state_dict(), os.path.join(save_path, "last_ckpt.pt"))
            writer.add_scalar("Loss/train", mean_train_loss, epoch)
            
            ###########################################################################################
            # After (eval_interval) epoch, go validating 
            if epoch == 1 or epoch % eval_interval == 0:
                val_losses = []
                total_correct = 0
                total = self.val_total
                
                for images, labels in tqdm(self.val_loader, desc="Validating"):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Training step get loss
                    with torch.no_grad():
                        # Calculate loss and accuracy
                        out = self.model(images)
                        # actual = self.one_hot_label(labels)
                        val_loss = self.loss_fn(out, labels)
                        correct = compare(out, labels)  
                        
                    val_losses.append(val_loss.item())    
                    total_correct += correct
                    
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
        
