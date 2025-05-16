import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

train_set = pd.read_csv('Data_Sets/train_energy_data.csv')
train_set = pd.get_dummies(train_set, columns=['Building Type', 'Day of Week'], drop_first=True)
pd.set_option('display.max_columns', None)


fig = plt.figure(figsize=(10,10))
sns.heatmap(train_set.corr(), vmin=-1, vmax=1, cmap='coolwarm', annot=True)
# plt.tight_layout()
# plt.show()

#Isolate numerical features and drop Target Feature
data_num = train_set.drop('Energy Consumption', axis = 1)
#Isolate label feature
data_label = train_set['Energy Consumption']

model = LinearRegression()
model.fit(data_num, data_label)

print("Omega values using scikit are:\n")
print(model.coef_)
print("")

#Find MSE on Training subset
pred_energy = model.predict(data_num)
mse_train = mean_squared_error(data_label, pred_energy)
print("MSE of Training set:", mse_train)
root_mse_train = np.sqrt(mse_train)
print("Root Mean Square Error of Training set:", root_mse_train)
print("")

#--TESTING WITH TEST SET#
test_set = pd.read_csv('Data_Sets/test_energy_data.csv')
test_set = pd.get_dummies(test_set, columns=['Building Type', 'Day of Week'], drop_first=True)
test_num = test_set.drop('Energy Consumption', axis = 1)
test_label = test_set['Energy Consumption']

pred_test_energy = model.predict(test_num)
mse_test = mean_squared_error(test_label, pred_test_energy)
print("MSE of Test set:", mse_test)
root_mse_test = np.sqrt(mse_test)
print("Root Mean Square Error of Test set:", root_mse_test)
r2_score = r2_score(test_label, pred_test_energy)
print("R2 Score of Test set:", r2_score)



