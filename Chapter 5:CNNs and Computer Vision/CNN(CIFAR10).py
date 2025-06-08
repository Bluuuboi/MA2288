import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import optuna

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914,0.4822, 0.4665], std=[0.2023,0.1994,0.2010])])

train_data = datasets.CIFAR10(root='./CIFAR10_data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./CIFAR10_data', train=False, download=True, transform=transform)

# print(train_data.data.shape)
# print(test_data.data.shape)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

for i, (train_images, train_labels) in enumerate(train_data):
    break
# print(train_images)
# print(train_images.shape)
# print(train_labels)
# print(train_labels.shape)

for i, (test_images, test_labels) in enumerate(test_data):
    break

# x = train_images.view(1,3,32,32)
# print(x.shape)
#
# x = F.relu(conv1(x))
# print(x.shape)
# x = F.max_pool2d(x, 2, 2)
# print(x.shape)
#
# x = F.relu(conv2(x))
# print(x.shape)
# x = F.max_pool2d(x, 2, 2)
# print(x.shape)

class ConvolutionalNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=8*8*64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 8*8*64)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

torch.manual_seed(1)
model = ConvolutionalNet()
# print(model)

#Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.022, momentum=0.9)

start = time.time()

epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for epoch in range(epochs):
    trn_corr = 0
    tst_corr = 0
    model.train()
    #Training model
    for b, (train_images, train_labels) in enumerate(train_loader):
        b+=1
        y_pred = model(train_images)
        loss = criterion(y_pred, train_labels)

        predicted = torch.max(y_pred, 1)[1]
        batch_corr = (predicted==train_labels).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 500 == 0:
            print(f'Epoch: {epoch} Batch:{b} Loss:{loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    #Test
    with torch.no_grad():
        for b, (test_images, test_labels) in enumerate(test_loader):
            y_val = model(test_images)
            predicted = torch.max(y_val, 1)[1]
            tst_corr += (predicted == test_labels).sum()

    loss = criterion(y_val, test_labels)
    test_losses.append(loss)
    test_correct.append(tst_corr)

end = time.time()
print(f'Training time: {end - start}')

print("Training loss: ", train_losses)
print("Test loss: ", test_losses)

train_losses = [tl.item() for tl in train_losses]
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

#Graph accuracy at end of each epoch
plt.plot([t/500 for t in train_correct], label='Training Accuracy')
plt.plot([t/100 for t in test_correct], label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    model.eval()
    correct = 0
    for test_images, test_labels in test_load_all:
        y_val = model(test_images)
        predicted = torch.max(y_val.data, 1)[1]
        correct += (predicted == test_labels).sum()

print(f'Test Accuracy: {100*correct.item()/len(test_data)}')


# Hyperparameter tuning
# Focus: Learning rate
# def objective(trial):
#     lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
#     torch.manual_seed(1)
#     model = ConvolutionalNet()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#
#     epochs = 5
#     for epoch in range(epochs):
#         model.train()
#         for b, (train_images, train_labels) in enumerate(train_loader):
#             y_pred = model(train_images)
#             loss = criterion(y_pred, train_labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         model.eval()
#         for test_images, test_labels in test_loader:
#             y_val = model(test_images)
#             predicted = torch.max(y_val.data, 1)[1]
#             correct += (predicted == test_labels).sum()
#             total += test_labels.size(0)
#
#     accuracy = correct/total
#     return accuracy
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)





