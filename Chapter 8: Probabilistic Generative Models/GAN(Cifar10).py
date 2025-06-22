#Reference: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import math
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()


batch_size = 16
mean=[0.4914,0.4822, 0.4665]
std=[0.2023,0.1994,0.2010]
torch.manual_seed(42)
transform = transforms.Compose([transforms.ToTensor()
                                   ,transforms.Normalize(mean=[0.4914,0.4822, 0.4665], std=[0.2023,0.1994,0.2010])])
train_data = datasets.CIFAR10(root='./CIFAR10_data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

real_images, cifar_labels = next(iter(train_loader))

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    img = real_images[i] * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)  # unnormalize
    plt.imshow(img.permute(1, 2, 0).numpy())
    # plt.imshow(real_images[i].permute(1, 2, 0).numpy())
    plt.axis('off')
plt.show()

#Implement the Discriminator
#Input = 3 channels * 32 pixels * 32 pixels
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=1), #(64*4*4)
            nn.Sigmoid()
        )
    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator()

#Implement the Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 1024), #64*4*4
            nn.LeakyReLU(),
            nn.Unflatten(1, (64,4,4)),
            #Upsample: 4x4 -> 8x8
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            #Upsample: 8x8 -> 16x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            #Upsample: 16x16 -> 32x32
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        output = self.model(x)
        return output
generator = Generator()

lr = 0.0001
epochs = 10
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
generator = generator.to(device)
discriminator = discriminator.to(device)


for epoch in range(epochs):
    for n, (real_images, _) in enumerate(train_loader):
        #Data for training the Discriminator
        real_images_labels = torch.ones(batch_size, 1).to(device)
        generated_images_labels = torch.zeros(batch_size, 1).to(device)
        latent_space_images = torch.randn((batch_size,100)).to(device)
        generated_images = generator(latent_space_images)
        all_images = torch.cat((real_images, generated_images))
        all_images_labels = torch.cat((real_images_labels, generated_images_labels))

        #Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_images)
        loss_discriminator = loss_function(output_discriminator, all_images_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        #Data for training the generator
        latent_space_images = torch.randn((batch_size,100)).to(device)

        #Training the generator
        generator.zero_grad()
        generated_images = generator(latent_space_images)
        output_discriminator_generated = discriminator(generated_images)
        loss_generator = loss_function(output_discriminator_generated, real_images_labels)
        loss_generator.backward()
        optimizer_generator.step()

        #Show loss
        if n==batch_size-1:
            print(f'Epoch: {epoch}, Loss D.: {loss_discriminator}')
            print(f'Epoch: {epoch}, Loss G.: {loss_generator}')

latent_space_samples = torch.randn((batch_size, 100))
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    image = generated_samples[i] * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)  # unnormalize
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')
plt.show()


#NOTE the formula for output size of convolutional layer (Downsampling)
#output size = 1+ ([input size + 2 * padding - kernel size]/[stride])

#Upsampling:
#output size = (input size - 1)*2 + kernel size - 2*padding