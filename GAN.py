import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_vector_size = 10):
        super(Generator, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
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

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 

    # You can modify the arguments of this function if needed
    def forward(self, z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        out = self.gen(z)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.disc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(3, 64, 4, 2, 1, bias=True),
            #REMOVE THIS BATCHNORM???
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
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
        
    # You can modify the arguments of this function if needed
    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        out = self.disc(x)
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
        
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

# You can modify the arguments of this function if needed
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
            
            
            #######################################################################
            #                       ** START OF YOUR CODE **
            #######################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

            # train with real
            model_D.zero_grad()
            data = data[0]
            #we add gaussian noise to the input because otherwise the discriminator always 'wins' (Actually workds well!!!) before : generator 
            #loss would exponentially increase 
            
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
            ####################################################################### 
            #                       ** END OF YOUR CODE **
            ####################################################################### 
            # Logging 
            if i % 50 == 0:
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(D_G_z=f"{D_G_z1:.3f}/{D_G_z2:.3f}", D_x=D_x,
                                  Loss_D=loss_D.item(), Loss_G=loss_G.item())

    if epoch == 0:
        save_image(denorm(real_data.cpu()).float(), content_path/'CW_GAN/real_samples.png')
    with torch.no_grad():
        fake = model_G(fixed_noise)
        save_image(denorm(fake.cpu()).float(), content_path/'CW_GAN/fake_samples_epoch_%03d.png')
        # % epoch
    train_losses_D.append(train_loss_D.cpu().detach().numpy() / len(loader_train))
    train_losses_G.append(train_loss_G.cpu().detach().numpy() / len(loader_train))
    plt.figure(figsize = (14,8))
    plt.plot(train_losses_D)
    plt.plot(train_losses_G)
    plt.title('Training losses')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['D Loss', 'G Loss'])
    plt.savefig(content_path/'CW_GAN/TrainingLoss.png')