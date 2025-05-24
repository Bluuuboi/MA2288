import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('../Data_Sets/Fish.csv')
data_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=30)

for train_index, test_index in data_split.split(df,df['Species']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

#Isolate numerical features(drop 'Species' feature) and drop target feature(Weight)
fishmarket_num = strat_train_set.drop({'Species','Weight'}, axis = 1)
#Isolate label feature
fishmarket_label = strat_train_set['Weight']
#Test data
fishmarket_test_num = strat_test_set.drop({'Species','Weight'}, axis = 1)
fishmarket_test_label = strat_test_set['Weight']

print('--RANDOM FOREST REGRESSOR--')
rand_forest_model = RandomForestRegressor(n_estimators=50, max_depth=3)
rand_forest_model.fit(fishmarket_num, fishmarket_label)
predicted_weight_training = rand_forest_model.predict(fishmarket_num)

mse_training = mean_squared_error(fishmarket_label, predicted_weight_training)
print('MSE of Training data using Random Forest Regressor is: ',mse_training)
sqrt_mse_training = np.sqrt(mse_training)
print('Root of MSE of Training data using Random Forest Regressor is: ', sqrt_mse_training)

#Testing with test data
predicted_weight_test = rand_forest_model.predict(fishmarket_test_num)

mse_test = mean_squared_error(fishmarket_test_label, predicted_weight_test)
print('MSE of Test data using Random Forest Regressor is: ', mse_test)
sqrt_mse_test = np.sqrt(mse_test)
print('Root of MSE of Test data using Random Forest Regressor is: ', sqrt_mse_test)

r2_score = r2_score(fishmarket_test_label, predicted_weight_test)
print('R^2 score using Random Forest Regressor is: ', r2_score)


