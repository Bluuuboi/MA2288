import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import tree


df = pd.read_csv('Data_Sets/Fish.csv')
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

print('--REGRESSION DECISION TREE--')
reg_tree_model = DecisionTreeRegressor(max_depth=3, min_samples_split=5)
reg_tree_model.fit(fishmarket_num, fishmarket_label)
predicted_weight_training_tree=reg_tree_model.predict(fishmarket_num)
mse_train_tree=mean_squared_error(fishmarket_label,predicted_weight_training_tree)

print('\nMean_squared_error on the training set using Regression Decision Tree=',mse_train_tree)

# find the square root of the mean squared error
root_mean_squared_error_train_tree=np.sqrt(mse_train_tree)
print('Root_mean_squared_error on the training set using Regression Decision Tree=',
      root_mean_squared_error_train_tree)

#Testing with test data
predicted_weight_test_tree=reg_tree_model.predict(fishmarket_test_num)

mse_test_tree=mean_squared_error(fishmarket_test_label,predicted_weight_test_tree)
print('\nMean_squared_error on the test set using Regression Tree=',mse_test_tree)

root_mean_squared_error_test_tree=np.sqrt(mse_test_tree)
print('Root_mean_squared_error on the test set using Regression Tree=',root_mean_squared_error_test_tree)

print('\n$R^2$ score=', r2_score(fishmarket_test_label,predicted_weight_test_tree))

fig = plt.figure(figsize=(25, 20))
tree_picture= tree.plot_tree(reg_tree_model, feature_names=fishmarket_num.columns, filled=False)
# plt.tight_layout()
# plt.show()
# plt.savefig("Fig_regression_tree.png")
print(reg_tree_model.feature_importances_)