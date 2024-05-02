import torch
import torch.nn as nn
import os
from torch.utils.data.dataset import random_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from IPython.display import clear_output; clear_output()
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, output_dim, hidden_dims=None, dropout_prob=0.0):
        super(Generator, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [125, 250, 500]  # Inverted order compared to Encoder

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

class Discriminator(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=None, dropout_prob=0.0):
        super(Discriminator, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [500, 250, 125]  # Same as Encoder

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        
        self.model = []

        for i in range(len(self.hidden_dims)):
            if i == 0:
                self.model.append(nn.Linear(self.input_dim + self.n_classes, self.hidden_dims[i]))
            else:
                self.model.append(nn.Linear(self.hidden_dims[i - 1] + self.n_classes, self.hidden_dims[i]))
            self.model.append(nn.LeakyReLU(0.2))
            self.model.append(nn.Dropout(dropout_prob))
        
        self.model.append(nn.Linear(self.hidden_dims[-1] + self.n_classes, 1))
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
        return x

class CGAN(nn.Module):
    def __init__(self, n_features, n_classes, latent_dim, hidden_dims=None):
        super(CGAN, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.generator = Generator(latent_dim, n_classes, n_features, hidden_dims)
        self.discriminator = Discriminator(n_features, n_classes, hidden_dims)

        
    def forward(self, z, labels):
        return self.generator(z, labels)

    def eval_model(self, num_samples, y):
        # y is already a one-hot encoded tensor indicating the class labels
        z = torch.randn(num_samples, self.generator.latent_dim).to(device)
        y = y.to(device)
        generated_samples = self.generator(z, y) # directly use the one-hot labels here
        return generated_samples

    def gradient_penalty(self, real_data, generated_data, labels):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).expand_as(real_data).to(device)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True).to(device)
        
        # Calculate discriminator's opinion on interpolated examples
        prob_interpolated = self.discriminator(interpolated, labels)
        
        gradients = torch.autograd.grad(outputs=prob_interpolated,
                                        inputs=interpolated,
                                        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                        create_graph=True,
                                        retain_graph=True)[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_model(self, data_loader, optimizer_G, optimizer_D, num_epochs):
        self.train()

        # Introduce a discriminator-to-generator update ratio
        D_update_ratio = 5

        # Early stopping initial values
        best_criteria = float('inf')
        epochs_no_improve = 0
        patience = 20
        
        D_loss = []
        G_loss = []

        for epoch in tqdm(range(num_epochs)):
            # Training Phase
            D_loss_epoch = []
            G_loss_epoch = []            
            for x_normalized, y in data_loader:
                x_normalized, y = x_normalized.to(device), y.to(device)
                batch_size = x_normalized.size(0)

                # Update the discriminator in a ratio
                for _ in range(D_update_ratio):
                    optimizer_D.zero_grad()

                    # Train on Real Data
                    prediction_real = self.discriminator(x_normalized, y)
                    
                    # Train on Fake Data
                    z = torch.randn(batch_size, self.generator.latent_dim).to(device)
                    gen_images = self.generator(z, y)


                    prediction_fake = self.discriminator(gen_images.detach(), y)

                    # Update discriminator
                    gradient_penalty = self.gradient_penalty(x_normalized, gen_images, y)
                    d_loss = -torch.mean(prediction_real) + torch.mean(prediction_fake) + 10 * gradient_penalty
                    d_loss.backward()
                    optimizer_D.step()

                optimizer_G.zero_grad()
                
                # Train Generator
                prediction_fake = self.discriminator(gen_images, y)
                g_loss = -torch.mean(prediction_fake)
                g_loss.backward()
                optimizer_G.step()
                
                D_loss_epoch.append(d_loss.item())
                G_loss_epoch.append(g_loss.item()) 
            
            D_loss.append(sum(D_loss_epoch) / len(D_loss_epoch))
            G_loss.append(sum(G_loss_epoch) / len(G_loss_epoch))         
            
            # Early stopping and model checkpointing
            stopping_criteria = abs(D_loss[-1])
            if stopping_criteria < best_criteria:
                best_criteria = stopping_criteria
                epochs_no_improve = 0
                best_model_weights = self.state_dict()  # save best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    #clear_output()
                    self.load_state_dict(best_model_weights)  # revert to best model
                    
                    plt.figure(figsize=(12,3))
                    plt.plot(G_loss,'g', label='Generator Loss')
                    plt.plot(D_loss,'r', label='Discriminator Loss')
                    plt.axvline(x=epoch-patience, color='k', linestyle='--', label='Early Stopping')
                    plt.grid(which='major', linestyle='--', linewidth='0.5', color='grey')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Generator and Discriminator Losses')
                    plt.legend()
                    
                    return
                
        plt.figure(figsize=(12,3))
        plt.plot(G_loss,'g', label='Generator Loss')
        plt.plot(D_loss,'r', label='Discriminator Loss')
        plt.axvline(x=epoch-patience, color='k', linestyle='--', label='Early Stopping')
        plt.grid(which='major', linestyle='--', linewidth='0.5', color='grey')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator and Discriminator Losses')
        plt.legend()
