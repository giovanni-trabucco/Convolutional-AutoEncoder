import torch
import numpy as np
from sklearn.model_selection import train_test_split 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class AutoEncoder(nn.Module):
    
    def __init__(self, encoded_space_dim, dropout = 0):
        super().__init__()
        
        self.dropout = nn.Dropout(p = dropout)
        self.act = nn.ReLU()
        
        self.train_loss_log = []
        self.val_loss_log = []
        
        
        self.encoder = nn.Sequential(
        
            ## CONVOLUTIONAL PART OF ENCODER
            # First convolutional layer
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, 
                      stride=2, padding=1),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, 
                      stride=2, padding=1),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, 
                      stride=2, padding=0),
            nn.ReLU(True),
            
            nn.Flatten(start_dim=1),
            
            
            ## LINEAR PART OF ENCODER
            # First linear layer
            nn.Linear(in_features=288, out_features=64),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(in_features=64, out_features=encoded_space_dim)
            
        )
        
        self.decoder = nn.Sequential(
            
            ## LINEAT PART OF DECODER
            # First linear layer
            nn.Linear(in_features=encoded_space_dim, out_features=64),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(in_features=64, out_features=288),
            nn.ReLU(True),
        
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            
            
            ## CONVOLUTIONAL PART OF DECODER
            # First transposed convolution
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, 
                               stride=2, output_padding=0),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, 
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, 
                               stride=2, padding=1, output_padding=1)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):   
        x = self.encode(x)
        x = self.decode(x)
        return x
    

        
    def train_and_validate(self, net_settings: dict, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader):
        loss_fn = net_settings["loss_fn"]
        lr = net_settings["lr"]
        optimizer = net_settings["optimizer"](self.parameters(), lr = lr)
        num_epochs = net_settings["epochs"]
        
        train_dataloader = train_dataloader
        val_dataloader = val_dataloader
        
        for epoch in range(num_epochs):
            print(f'Epoch #{epoch+1}')
            
            #TRAINING STEP
            self.train()
            loss_batches = []
            for sample_batched in train_dataloader:
                images = sample_batched[0].to(device)
                out = self(images)
                loss = loss_fn(out, images)
                self.zero_grad()
                loss.backward()
                optimizer.step()
                loss_batches.append(loss.detach().cpu().numpy())
            
            loss_epoch = np.mean(loss_batches)
            print(f'Average TRAINING loss for this epoch: {loss_epoch}')
            self.train_loss_log.append(loss_epoch)
            
            #VALIDATION STEP
            self.eval()
            loss_batches = []
            with torch.no_grad():
                for sample_batched in val_dataloader:
                    images = sample_batched[0].to(device)
                    out = self(images)
                    loss = loss_fn(out, images)
                    loss_batches.append(loss.detach().cpu().numpy())
                
            loss_epoch = np.mean(loss_batches)
            print(f'Average VALIDATION loss for this epoch: {loss_epoch}')
            self.val_loss_log.append(loss_epoch)
            
            ### Plot progress
            # Get the output of a specific image (the test image at index 0 in this case)
            test_dataset = test_dataloader.dataset
            img = test_dataset[0][0].unsqueeze(0).to(device)
            self.eval()
            with torch.no_grad():
                rec_img  = self(img)
            # Plot the reconstructed image
            fig, axs = plt.subplots(1, 2, figsize=(6,3))
            axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[0].set_title('Original image')
            axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[1].set_title('Reconstructed image (EPOCH %d)' % (epoch + 1))
            plt.tight_layout()
            plt.pause(0.1)
            plt.show()
            plt.close()
            