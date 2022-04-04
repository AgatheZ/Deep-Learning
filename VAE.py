import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms

# Necessary Hyperparameters 
num_epochs = 40
learning_rate = 0.0001
batch_size = 64
latent_dim = 10

transform = transforms.Compose([
    transforms.ToTensor(),
])

def denorm(x):
    return x
    
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.z_dim = 28//2**2

        # fully connected layers for learning representations
        self.fc_mu = nn.Linear(self.z_dim**2 * 32, latent_dim)
        self.fc_log_var = nn.Linear(self.z_dim**2 *32, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.z_dim**2 * 32)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding = 1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding =1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding = 1),
            nn.Sigmoid()
         
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
    def reparametrize(self, mu, logvar): #REPARAMETRIZATION TRICK: normal distributions are just scaled and translated 
        #from normal distributions 
        std = torch.exp(0.5*logvar) 
        eps = torch.randn_like(std) #random noise
        return mu + std*eps

    def decode(self, z):
        z = self.fc2(z)
        dec = z.view(z.size(0), 32, self.z_dim, self.z_dim)
        dec = self.decoder(dec)
        return dec 
    
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        output = self.decode(z)
        return output, mean, logvar
        

model = VAE(latent_dim)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
print(model)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)