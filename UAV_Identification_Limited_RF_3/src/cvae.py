import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data.dataset import random_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from IPython.display import clear_output; clear_output()
from tqdm import tqdm
from math import cos, pi
from .utils import stratified_split_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Encoder(nn.Module):
    def __init__(self, input_dim, n_classes, latent_dim, hidden_dims=None, dropout_prob=0.0):
        super(Encoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [500, 250, 125]

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        self.model = []

        for i in range(len(self.hidden_dims)):
            if i == 0:
                self.model.append(nn.Linear(self.input_dim + self.n_classes, self.hidden_dims[i]))
            else:
                self.model.append(nn.Linear(self.hidden_dims[i - 1] + self.n_classes, self.hidden_dims[i]))
            self.model.append(nn.LeakyReLU(0.2))
            self.model.append(nn.Dropout(dropout_prob))
        
        self.model.append(nn.Linear(self.hidden_dims[-1] + self.n_classes, 2 * self.latent_dim))
        self.model = nn.Sequential(*self.model)
        
        self.apply(self.init_weights)
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x, labels):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                x = torch.cat([x, labels], dim=1)
            x = layer(x)
        mu = x[:, :self.latent_dim]
        log_var = x[:, self.latent_dim:]
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_classes, output_dim, hidden_dims=None, dropout_prob=0.0):
        super(Decoder, self).__init__()
        
                
        if hidden_dims is None:
            hidden_dims = [125, 250, 500]

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        self.model = []

        for i in range(len(self.hidden_dims)):
            if i == 0:
                self.model.append(nn.Linear(self.latent_dim + self.n_classes, self.hidden_dims[i]))
            else:
                self.model.append(nn.Linear(self.hidden_dims[i - 1] + self.n_classes, self.hidden_dims[i]))
            self.model.append(nn.LeakyReLU(0.2))
            self.model.append(nn.Dropout(dropout_prob))
        
        self.model.append(nn.Linear(self.hidden_dims[-1] + self.n_classes, self.output_dim))
        self.model.append(nn.Tanh())
        self.model = nn.Sequential(*self.model)
        
        self.apply(self.init_weights)
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
            
    def forward(self, z, labels):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                z = torch.cat([z, labels], dim=1)
            z = layer(z)
        return z

    
class CVAE(nn.Module):
    def __init__(self, n_features, n_classes, latent_dim, hidden_dims=None):
        super(CVAE, self).__init__() 

        self.n_features = n_features
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.encoder = Encoder(n_features, n_classes, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, n_classes, n_features, hidden_dims)
        self.labels = []

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, labels):
        mu, log_var = self.encoder(x, labels)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z, labels), mu, log_var

    def eval_model(self, num_samples, y):
        z = torch.randn(num_samples, self.latent_dim).to(device)  # Corrected latent_dim reference
        y = y.to(device)
        generated_samples = self.decoder(z, y)  # Changed to use decoder
        return generated_samples
    
    def compute_loss(self, recon_x, x, mu, log_var, kl_weight=1.0):
        # Reconstruction loss (e.g., MSE or BCE)
        #recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        recon_loss = F.l1_loss(recon_x, x, reduction='sum')

        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        return recon_loss + kl_weight*kl_loss
    
  

    def train_model(self, data_loader, optimizer, num_epochs):
        self.train()

        # Split the dataset into training and validation sets
        validation_split = 0.25
        train_loader, val_loader = stratified_split_loader(data_loader, validation_split)

        # Early stopping initial values
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 20
        
        # Cyclic Annealing schedule parameters
        def cyclical_annealing_step(epoch, cycle_length=10, max_value=1.0):
            """Compute the KL weight for the current epoch using a cyclical schedule."""
            phase = epoch % cycle_length
            return max_value * (1 - cos(pi * phase / cycle_length)) / 2 
        cycle_length = 10  # Number of epochs in a cycle
        max_kl_weight = 1 # Maximum KL weight
        
        # Initialize lists to track losses
        training_losses = []
        validation_losses = []
        
        for epoch in tqdm(range(num_epochs)):
            kl_weight = cyclical_annealing_step(epoch, cycle_length=cycle_length, max_value=max_kl_weight)
    
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # Forward pass
                recon_x, mu, log_var = self(x, y)

                # Compute loss
                loss = self.compute_loss(recon_x, x, mu, log_var, kl_weight=kl_weight)

                # Zero the gradients and perform backward pass
                optimizer.zero_grad()
                loss.backward()

                # Update weights
                optimizer.step()

                total_loss += loss.item()

            # Average loss for this epoch
            avg_loss = total_loss / len(train_loader.dataset)
            
            training_losses.append(avg_loss)
            
            # Validation Phase
            self.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    recon_x_val, mu_val, log_var_val = self(x_val, y_val)
                    val_loss = self.compute_loss(recon_x_val, x_val, mu_val, log_var_val, kl_weight=kl_weight)
                    val_losses.append(val_loss.item())

            average_val_loss = sum(val_losses) / len(val_losses)
            validation_losses.append(average_val_loss)
                
            # Early stopping and model checkpointing
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                epochs_no_improve = 0
                best_model_weights = self.state_dict()  # save best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    self.load_state_dict(best_model_weights)  # revert to best model

                    # Plot training and validation losses at early stopping
                    plt.figure(figsize=(12, 3))
                    plt.plot(training_losses, 'g', label='Training Loss')
                    plt.plot(validation_losses, 'b', label='Validation Loss')
                    plt.axvline(x=epoch-patience, color='k', linestyle='--', label='Early Stopping')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Losses')
                    plt.legend()
                    plt.grid(which='major', linestyle='--', linewidth='0.5', color='grey')

                    return

            self.train()  # Set back to training mode for next epoch


        print("Training complete.")