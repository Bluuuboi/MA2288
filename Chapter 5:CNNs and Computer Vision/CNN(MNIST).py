import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import time

#Transform MNIST dataset to 4-dim Tensor(# of images, Height, Width, Color Channel)
transform = transforms.ToTensor()

#Train data
train_data = datasets.MNIST(root='./cnn_data', train=True, download=True, transform=transform)
#Test data
test_data = datasets.MNIST(root='./cnn_data', train=False, download=True, transform=transform)

# print(train_data)
# print(test_data)

#Create a small batch size for images, i.e. 10
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

#Small example
conv1 = nn.Conv2d(1,6,3,1)
conv2 = nn.Conv2d(6,16,3,1)

#Grab one MNIST image
for i, (X_train, y_train) in enumerate(train_data):
    break
# print(X_train)
# print(X_train.shape)

for i, (X_test, y_test) in enumerate(test_data):
    break

x = X_train.view(1,1,28,28)
# print(x)
print(x.shape)


#Perform first convolution
#ReLU for activation function
x = F.relu(conv1(x))
# print(x)
print(x.shape)
#Result: 1 is the single image, 6 is the filters/feature maps, 26x26 is the image size
#No need to pad given the MNIST Image

#Pass through the pooling layer
x = F.max_pool2d(x,2,2)
print(x.shape)
#Result: 1,6,13,13
#Since 26/2=13

#Second convolutional layer
x = F.relu(conv2(x))
print(x.shape)

#Pass through second pooling layer
x = F.max_pool2d(x,2,2)
print(x.shape)
#Result 1,16,5,5

#Define CNN model
#2 convolutional layers in this case
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        #Fully connected layer
        #120,84 is arbitrary
        #End with 10 since thats the number of classes we have
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)

        #Re-View to flatten it out to send it to ANN layer
        x = x.view(-1, 5*5*16) #-1 so we can vary the batch size

        #Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

#Instantiate our model
torch.manual_seed(41)
model = ConvolutionalNetwork()
# print(model)

#Loss function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start = time.time()

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for epoch in range(epochs):
    trn_corr = 0
    tst_corr = 0

    #Train
    for b,(X_train,y_train) in enumerate(train_loader):
        b+=1 #start our batches at 1
        y_pred = model(X_train) #Get predicted values from training data, results not flattened.
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1] #Add up the number of correct predictions. Indexed off the first point.
        batch_corr = (predicted==y_train).sum() #No. we get correct from this batch
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 600 == 0:
            print(f'Epoch: {epoch} Batch:{b} Loss:{loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)


    #Test
    with torch.no_grad():
        for b,(X_test,y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)


end = time.time()
print("Training time: ", end - start)
print("Training loss: ", train_losses)
print("Test loss: ", test_losses)

#Graph the loss at each epoch
train_losses = [tl.item() for tl in train_losses]
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

#Graph accuracy at end of each epoch
plt.plot([t/600 for t in train_correct], label='Training Accuracy')
plt.plot([t/100 for t in test_correct], label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val.data, 1)[1]
        correct += (predicted == y_test).sum()

print(f'Test Accuracy: {100*correct.item()/len(test_data)}')