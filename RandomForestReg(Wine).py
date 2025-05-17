import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('Data_Sets/winequality-red.csv')
# print(df.head())

data_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in data_split.split(df,df['quality']):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

#Drop target feature(quality)
wine_train_num = train_set.drop('quality', axis=1)
wine_train_label = train_set['quality']

#Do the same for test data
wine_test_num = test_set.drop('quality', axis=1)
wine_test_label = test_set['quality']

rand_forest_model = RandomForestRegressor(n_estimators=10, max_depth=3)
rand_forest_model.fit(wine_train_num, wine_train_label)
predicted_wine_train = rand_forest_model.predict(wine_train_num)

mse_train = mean_squared_error(wine_train_label, predicted_wine_train)
print('MSE of Training data using Random Forest:', mse_train)
sqrt_mse_train = np.sqrt(mse_train)
print('Root of MSE of Training data using Random Forest:', sqrt_mse_train)

r2score_train = r2_score(wine_train_label, predicted_wine_train)
print('R2 of Training data using Random Forest:', r2score_train)

#Testing with test data
predicted_wine_test = rand_forest_model.predict(wine_test_num)

mse_test = mean_squared_error(wine_test_label, predicted_wine_test)
print('MSE of Test data using Random Forest:', mse_test)
sqrt_mse_test = np.sqrt(mse_test)
print('Root of MSE of Test data using Random Forest:', sqrt_mse_test)

r2score_test = r2_score(wine_test_label, predicted_wine_test)
print('R2 of Test data using Random Forest:', r2score_test)

#Model is overfitted here, data is not cleaned up before usage