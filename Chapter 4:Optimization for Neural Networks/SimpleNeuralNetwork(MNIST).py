import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import optuna
import time
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
from sklearn.datasets import fetch_openml

mnist_data = fetch_openml('mnist_784', version=1, data_home='Data_Sets/MNIST')

x,y_true = mnist_data['data'],mnist_data['target']
y_true = y_true.astype(np.uint8)

#Standard scaling
scaler = StandardScaler()
x_train,x_test,y_true_train,y_true_test = x.iloc[0:60000,:],x.iloc[60000:70000,:],y_true.iloc[0:60000],y_true.iloc[60000:70000]
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


y_true_train_8, y_true_test_8=(y_true_train==8), (y_true_test==8)
# print(y_true_train_8)

x_train_tensor = torch.FloatTensor(x_train_scaled)
x_test_tensor = torch.FloatTensor(x_test_scaled)

#unsqueeze is to make the output dimensions same as the input dimensions
y_train_tensor = torch.FloatTensor(y_true_train_8.to_numpy()).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_true_test_8.to_numpy()).unsqueeze(1)

# print(y_train_tensor)

# 28 by 28 pixels = 784 input features ->
# Hidden layer 1, 64 neurons ->
# Hidden layer 2, 64 neurons ->
# Output, 1 neuron, since we are only doing binary classification
class Model(nn.Module):
    def __init__(self, in_features = 784,h1=64,h2=64,out_features=1):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

torch.manual_seed(42)
model = Model()
lossFunction = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
epochs = 100
start = time.time()
for epoch in range(epochs):
    model.train()
    y_pred_train = model(x_train_tensor)
    loss = lossFunction(y_pred_train, y_train_tensor)

    #Zero the gradients stored within model's .grad parameters
    optimizer.zero_grad()
    #Compute gradient of loss function wrt to model parameters
    loss.backward()
    #Update model parameters
    optimizer.step()
    #Print loss
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

end = time.time()
print('Time taken to train model:', end-start)

model.eval()
with torch.no_grad():
    y_pred_test = model(x_test_tensor)
    y_pred_label = (torch.sigmoid(y_pred_test) > 0.5).float()
    accuracy = (y_pred_label == y_test_tensor).float().mean()
print('Test accuracy:', accuracy.item())

print(y_true_test_8.to_numpy())
print(y_pred_test)

#Note that I did not bother to actually get the optimal values for the hyperparameters
#but I did run the code below and it works
#Tune hyperparameters
start = time.time()
def objective(trial):
    lr = trial.suggest_float("lr", 0.001, 1e-1, log=True)
    h1 = trial.suggest_int('h1', 64, 128)
    h2 = trial.suggest_int('h2', 64, 128)
    torch.manual_seed(42)
    model = Model(h1= h1, h2= h2)
    lossFunction = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    epochs = 100

    for epoch in range(epochs):
        model.train()
        y_pred_train = model(x_train_tensor)
        loss = lossFunction(y_pred_train, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test_tensor)
        y_pred_label = (torch.sigmoid(y_pred_test) > 0.5).float()
        correct = (y_pred_label == y_test_tensor).float().sum().item()

        accuracy = correct / len(y_test_tensor)
        return accuracy  # Optuna will maximize this

# Run the optimization
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print(f"  Params: {trial.params}")

end = time.time()
print("Time taken for hyperparameter tuning:",end - start)