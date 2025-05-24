import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
# from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import optuna


#Create a Model class that inherits nn.Module
class Model(nn.Module):
    #Input layer (4 features of the flower) ->
    #Hidden layer 1 (arbitrary number of neurons) ->
    #Hidden layer 2 (arbitrary number of neurons) ->
    #Output layer (3 species of flowers, represented by 4 neurons)
    def __init__(self, in_features=4, h1=22, h2=25, out_features=3):
        super(Model, self).__init__() #Instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

#Pick manual seed for randomization
torch.manual_seed(30)
#Create an instance of the model
model = Model()

#Load data
df = pd.read_csv('../Data_Sets/Iris (1).csv')
# print(df.head())
# print(df.describe())

#Change Species to Number
df['Species'] = df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
df = df.drop('Id', axis=1)
# print(df.head())

#Correlation heatmap
# plt.figure(figsize=(10,10))
# corr = df.corr()
# sns.heatmap(corr, annot=True)
# plt.show()

#Isolate label and change both dataframes to arrays
y = df['Species'].values
x = df.drop('Species', axis=1).values
# print(x.head())
# print(y)

#Split data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
# print(x_train.head())
# print(x_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print(y_test)

#Convert x features to FloatTensor and y labels to LongTensor respectively
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set Loss function
lossFunction = nn.CrossEntropyLoss()

#Choose our optimizer, in this case Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.04228777485457357)

#Train our model
epochs = 100
losses_arr = []
for epoch in range(epochs):
    model.train()
    y_pred_train = model(x_train)
    loss = lossFunction(y_pred_train, y_train)

    #Append losses to array
    losses_arr.append(loss.item())

    #Zero the gradients stored within model's .grad parameters
    optimizer.zero_grad()
    #Compute gradient of loss function wrt to model parameters
    loss.backward()
    #Update model parameters
    optimizer.step()

    #Print loss
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#Plot the losses
plt.plot(range(epochs), losses_arr)
plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.show()

correct = 0
model.eval()
with torch.no_grad():
    y_pred_test = model(x_test)
    predicted_classes = torch.argmax(y_pred_test, dim=1)
    for i in range(y_pred_test.size(0)):
        if predicted_classes[i] == y_test[i]:
            correct += 1
    accuracy = correct / y_pred_test.size(0)

print(y_pred_test)
print("Predicted labels:", predicted_classes.tolist())
print("Actual labels:   ", y_test.tolist())
print('Correct:', correct)
print('Accuracy:', accuracy)


#Using Optuna to tune learning rate#
def objective(trial):
    lr = trial.suggest_float('lr', 0.001, 1e-1, log=True)
    h1 = trial.suggest_int('h1', 10, 32)
    h2 = trial.suggest_int('h2', 10, 32)
    torch.manual_seed(30)
    model = Model(h1=h1,h2=h2)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        y_pred_train = model(x_train)
        loss = lossFunction(y_pred_train, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate accuracy on test set
    correct = 0
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)
        predicted_classes = torch.argmax(y_pred_test, dim=1)
        correct = (predicted_classes == y_test).sum().item()

    accuracy = correct / len(y_test)
    return accuracy  # Optuna will maximize this

# Run the optimization
# study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=30))
# study.optimize(objective, n_trials=200)

# print("Best trial:")
# trial = study.best_trial
# print(f"  Accuracy: {trial.value}")
# print(f"  Params: {trial.params}")




##

##TEST THIS OUT WHEN SKORCH SUPPORTS PYTHON 3.13##
#Tuning hyperparameter, eta
# net = NeuralNetClassifier(
#     module=Model,
#     module__in_features=4,
#     module__h1=10,
#     module__h2=10,
#     module__out_features=3,
#     max_epochs=300,
#     optimizer=torch.optim.SGD,
#     criterion=nn.CrossEntropyLoss,
#     verbose=0
# )
#
# params_grid = {'lr': [0.0001, 0.001, 0.01, 0.05, 0.1, 1]}
# grid_search = GridSearchCV(net, param_grid=params_grid, scoring='accuracy', cv=5, refit=True)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)
####


