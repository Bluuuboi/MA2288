import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import tree

df = pd.read_csv('Data_Sets/winequality-red.csv')
# print(df.head())

data_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in data_split.split(df,df['quality']):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

#Checking proportion of data set
# print(train_set['quality'].value_counts()/len(train_set))
# print(df['quality'].value_counts()/len(df))

#Drop target feature(quality)
winedata_train_num = train_set.drop('quality', axis=1)
winedata_train_label = train_set['quality']
#Do the same for test data
winedata_test_num = test_set.drop('quality', axis=1)
winedata_test_label = test_set['quality']

reg_tree = DecisionTreeRegressor(max_depth=3, min_samples_split=5)
reg_tree.fit(winedata_train_num, winedata_train_label)
predicted_wine_training = reg_tree.predict(winedata_train_num)

#MSE and root of MSE for training
mse_train = mean_squared_error(winedata_train_label, predicted_wine_training)
print('MSE train is: ', mse_train)
root_mse_train = np.sqrt(mse_train)
print('Root Mean Square Error = ', root_mse_train)

r2score = r2_score(winedata_train_label,predicted_wine_training)
print(r2score)

#Testing test data
predicted_wine_test = reg_tree.predict(winedata_test_num)

mse_test = mean_squared_error(winedata_test_label, predicted_wine_test)
print('--TEST DATA--')
print('MSE test is: ', mse_test)
root_mse_test = np.sqrt(mse_test)
print('Root Mean Square Error = ', root_mse_test)
r2_test = r2_score(winedata_test_label, predicted_wine_test)
print(r2_test)



