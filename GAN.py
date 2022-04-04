import numpy as np
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
import torch
import tqdm
import torch.nn as nn

GPU = True # Choose whether to use GPU
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f'Using {device}')

mean = torch.Tensor([0.4914, 0.4822, 0.4465])
std = torch.Tensor([0.247, 0.243, 0.261])
num_epochs = 5
batch_size = 64
learning_rate = 0.00001
latent_vector_size = 100

transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std),                        
])
# note - data_path was initialized at the top of the notebook
cifar10_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=batch_size)
loader_test = DataLoader(cifar10_test, batch_size=batch_size)

class Generator(nn.Module):
    def __init__(self, latent_vector_size = 10):
        super(Generator, self).__init__()
        self.latent_vector_size = latent_vector_size
        self.gen = nn.Sequential(

            nn.ConvTranspose2d(self.latent_vector_size, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.gen(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(3, 64, 4, 2, 1, bias=True),

            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
      
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.5),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
    
            nn.Conv2d(512, 1, 4, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.disc(x)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

use_weights_init = True

model_G = Generator().to(device)
if use_weights_init:
    model_G.apply(weights_init)
params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
print("Total number of parameters in Generator is: {}".format(params_G))
print(model_G)
print('\n')

model_D = Discriminator().to(device)
if use_weights_init:
    model_D.apply(weights_init)
params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
print("Total number of parameters in Discriminator is: {}".format(params_D))
print(model_D)
print('\n')

print("Total number of parameters is: {:,}".format(params_G + params_D))


def loss_function(out, label):
    loss = nn.BCELoss()
    return loss(out, label)

beta1 = 0.5
optimizerD = torch.optim.Adam(model_D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))
fixed_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
# Additional input variables should be defined here
train_losses_G = []
train_losses_D = []

# <- You may wish to add logging info here
for epoch in range(num_epochs):
    # <- You may wish to add logging info here
    train_loss_D = 0
    train_loss_G = 0
    with tqdm.tqdm(loader_train, unit="batch") as tepoch: 
        for i, data in enumerate(tepoch):

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # train with real
            model_D.zero_grad()
            data = data[0]
            #we add gaussian noise to the input because otherwise the discriminator always 'wins' 
            
            real_data = (data +  max(0.000,(1 - epoch*0.1))*(0.1**0.5)*torch.randn(data.shape)).to(device)
            real_data = data.to(device)
            
            random_true_label = np.random.uniform(0.7, 1)
            true_label = torch.full((data.shape[0],), random_true_label, dtype=torch.float, device=device)
            true_output = model_D(real_data).view(-1)
            
           
            
            loss_D_real = loss_function(true_output, true_label)
            loss_D_real.backward()
            D_x = true_output.mean().item()
            
            # train with fake
            noise = torch.randn(data.shape[0], latent_vector_size, 1, 1, device=device) #sample from gaussian??
            fake_data = model_G(noise) +  max(0.000,(1 - epoch*0.1))*(0.1**0.5)*torch.randn(data.shape).to(device)
            fake_data = model_G(noise)
            
            random_fake_label = np.random.uniform(0, 0.3)
            fake_label = torch.full((data.shape[0],), random_fake_label, dtype=torch.float, device=device)
            fake_output = model_D(fake_data.detach()).view(-1)
            loss_D_fake = loss_function(fake_output, fake_label)
            loss_D_fake.backward()
            D_G_z1 = fake_output.mean().item()
            
            
            loss_D = loss_D_fake + loss_D_real 
            
            optimizerD.step()
            
            # (2) Update G network: maximize log(D(G(z)))
            model_G.zero_grad()
            random_true_label = np.random.uniform(0.7, 1)
            G_label = torch.full((data.shape[0],), random_true_label, dtype=torch.float, device=device)
            fake_data_noisy = (fake_data)
            G_output = model_D(fake_data).view(-1)
            
            loss_G = loss_function(G_output, G_label)
            loss_G.backward()
            D_G_z2 = G_output.mean().item()
            optimizerG.step()
            
            train_loss_D += loss_D
            train_loss_G += loss_G
 
            # Logging 
            if i % 50 == 0:
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(D_G_z=f"{D_G_z1:.3f}/{D_G_z2:.3f}", D_x=D_x,
                                  Loss_D=loss_D.item(), Loss_G=loss_G.item())

    
    train_losses_D.append(train_loss_D.cpu().detach().numpy() / len(loader_train))
    train_losses_G.append(train_loss_G.cpu().detach().numpy() / len(loader_train))
   