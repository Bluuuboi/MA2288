import torch
import torch.nn.functional as F
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import optuna

df = pd.read_csv('../Data_Sets/Fish.csv')

#Goal: Predict weight of the fish via all the other factors

data_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=30)
for train_index, test_index in data_split.split(df,df['Species']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# plt.figure(figsize=(10,10))
# corr = df.drop('Species', axis=1).corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()

#Isolate features and labels
scaler = StandardScaler()
fm_num_train = strat_train_set.drop({'Species','Weight'},axis=1)
fm_num_test = strat_test_set.drop({'Species','Weight'},axis=1)
fm_num_train = scaler.fit_transform(fm_num_train)
fm_num_test = scaler.transform(fm_num_test)

#Convert DataFrames in arrays and then into Tensors
fm_num_train_tensor = torch.FloatTensor(fm_num_train)
fm_num_test_tensor = torch.FloatTensor(fm_num_test)
fm_label_train_tensor = torch.FloatTensor(strat_train_set['Weight'].values).unsqueeze(1)
fm_label_test_tensor = torch.FloatTensor(strat_test_set['Weight'].values).unsqueeze(1)
# print(fm_num_train_tensor.shape)
# print(fm_label_train_tensor.shape)


class Model(nn.Module):
    def __init__(self, input = 5, h1 = 124, h2 = 126, output = 1):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

torch.manual_seed(30)
model = Model()
lossFunction = nn.MSELoss()
#Using Adam algo as optimizer this time
optimizer = torch.optim.Adam(model.parameters(), lr=0.007033231118706293)

epochs = 100
for epoch in range(epochs):
    model.train()
    y_pred_train = model(fm_num_train_tensor)
    loss = lossFunction(y_pred_train, fm_label_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    y_pred_test = model(fm_num_test_tensor)
    test_loss = lossFunction(y_pred_test, fm_label_test_tensor)
    print(f'Test Loss: {test_loss.item()}')

plt.figure(figsize=(10,10))
plt.scatter(fm_label_test_tensor.numpy(), y_pred_test.numpy(), alpha=0.6)
plt.xlabel('Actual Weight')
plt.ylabel('Predicted Weight')
plt.title('Predicted vs Actual Fish Weight')
plt.plot([0, max(fm_label_test_tensor.max(), y_pred_test.max())], [0, max(fm_label_test_tensor.max(), y_pred_test.max())], 'r--')
plt.grid(True)
plt.show()

#Hyperparameter tuning
# def objective(trial):
#     lr = trial.suggest_float('lr', 0.000001, 0.01, log=True)
#     h1 = trial.suggest_int('h1', 1, 128)
#     h2 = trial.suggest_int('h2', 1, 128)
#     model = Model(h1= h1, h2= h2)
#     lossFunction = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     epochs = 100
#     for epoch in range(epochs):
#         model.train()
#         y_pred_train = model(fm_num_train_tensor)
#         loss = lossFunction(y_pred_train, fm_label_train_tensor)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # Evaluation on test set
#     model.eval()
#     with torch.no_grad():
#         y_pred_test = model(fm_num_test_tensor)
#         test_loss = lossFunction(y_pred_test, fm_label_test_tensor)
#
#     return test_loss.item()
#
# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=30))
# study.optimize(objective, n_trials=5000)
# print("Best trial: ", study.best_params)