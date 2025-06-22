#Reference: https://realpython.com/generative-adversarial-networks/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import math

torch.manual_seed(42)


#Preparing training data
#Training data is composed of 1024 pairs (x1,x2), so that x2 consists of the values of the sine of x1 for x1 in the range of 0 to 2pi
train_data_length = 1024
#Initialize train_data, a tensor with dimensions 1024x2, all entries zero.
train_data = torch.zeros((train_data_length,2))
#Use first col of train_data to store random values in the interval from 0 to 2pi
train_data[:,0]= 2 * math.pi * torch.rand(train_data_length)
#Use second col of train_data to store the sine of values of the first col
train_data[:,1] = torch.sin(train_data[:,0])
#Create a tensor of labels, which is required by pytorch's DataLoader
train_labels = torch.zeros(train_data_length)
#Create train_set as a list of tuples, with each row of train_data and train_labels represented in each tuple
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

#Now we plot each data point for visualization
plt.plot(train_data[:,0], train_data[:,1], ".")
plt.show()

#Now, we create a dataloader called train_loader, which will shuffle the data from train_set and return batches of 32 samples that'll be used to train the NN
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

#Implementing the Discriminator
#For this case, the input will be two-dimensional, and the output one-dimensional
#After first,second and third hidden layers, use Dropout to prevent overfitting
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            #Input is two-dimensional, and the first hidden layer is composed of 256 neurons with ReLU activation
            nn.Linear(2,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            #Second and third layers are composed of 128 and 64 neurons respectively
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            #Output is composed of a single neuron with sigmoidal activation to represent a probability
            nn.Linear(64,1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator()

#Implementing the Generator
#Two hidden layers with 16 and 32 neurons respectively
#Input are two random values from train_data above
#Output consists of a vector of two elements with values ranging from negative infinity to infinity
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,2),
        )
    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

#Training the model
lr = 0.001
epochs = 300
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

#Training loop
for epoch in range(epochs):
    #Get real samples of the current batch from dataloader and assign them to real_samples
    for n, (real_samples, _) in enumerate(train_loader):
        #Data for training the discriminator
        #Create labels with the value 1 for the real samples, then assign the labels to real_samples_labels
        real_samples_labels = torch.ones((batch_size, 1))
        #Create the generated samples by storing random data in latent_space samples, which is then fed to the generator to obtain generated_samples
        latent_space_samples = torch.randn(batch_size, 2)
        generated_samples = generator(latent_space_samples)
        #Value 0 is assigned to labels for the generated samples, and the values are stored in generated_samples_labels
        generated_samples_labels = torch.zeros((batch_size, 1))
        #Concatenate the real and generated samples and labels and store them in all_samples/all_samples_labels to be used to train the discriminator
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        #Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        #Data for training the generator
        #Store random data in latent_space_samples, with number of lines equal to batch_size
        #Two columns used here, since input is two-dimensional
        latent_space_samples = torch.randn((batch_size, 2))

        #Training the generator
        generator.zero_grad()
        #Feed generator with latent_space_samples and store its output in generated_samples
        generated_samples = generator(latent_space_samples)
        #Feed generated_samples(i.e. generator's output) into the discriminator and store its output in output_discriminator_generated, which will be used as the output of the whole model
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        #Show loss
        if epoch%10==0 and n==batch_size-1:
            print(f'Epoch: {epoch}, Loss D.: {loss_discriminator}')
            print(f'Epoch: {epoch}, Loss G.: {loss_generator}')


#Checking the samples generated by the GAN
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:,0], generated_samples[:,1], ".")
plt.show()

#IN THIS CASE,a number close to 1 will be generated if sample is classified as coming from training data
#a number close to 0 will be generated if sample is classified as coming from generator
#If discriminator gets the classification correct, loss will be low. Vice versa
#If generator loss is low, it managed to trick the discriminator. Vice versa
