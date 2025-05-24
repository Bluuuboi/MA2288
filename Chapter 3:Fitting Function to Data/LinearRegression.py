import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn as sk
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv('../Data_Sets/Fish.csv')
print(df.describe())
species_num = df['Species'].value_counts()

# df.hist(bins=50,figsize=(10,10))
# plt.tight_layout()
# plt.show()

#Splitting data into Training and Test Subsets
data_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=30)

for train_index, test_index in data_split.split(df,df['Species']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

print('Proportions of fish species in the test data subset:')
print(strat_test_set['Species'].value_counts())
print(strat_test_set['Species'].value_counts()/len(strat_test_set))
print("\n")

print('Proportions of fish species in the original data set:')
print(species_num)
print(species_num/len(df))
print("\n")

#Correlation Matrix
corr_matrix = strat_train_set.select_dtypes(include=[np.number]).corr()
print("Correlation matrix:\n" , corr_matrix)
print("\n")
#Check the correlation of each length feature with Weight(i.e. the target feature)
print("Correlation of each length feature with Weight:\n" , corr_matrix['Weight'])
print("\n")

#Scatter matrix of correlation
scatter_matrix(strat_train_set, figsize = (17,17), color = 'black')
# plt.tight_layout()
# plt.show()

#Isolate numerical features(drop 'Species' feature) and drop target feature(Weight)
fishmarket_num = strat_train_set.drop({'Species','Weight'}, axis = 1)
#Isolate label feature
fishmarket_label = strat_train_set['Weight']

model = LinearRegression()
model.fit(fishmarket_num, fishmarket_label)

#Print the omegas, i.e. the w's
print("Omega values using scikit are:\n")
print("Bias Term:" , model.intercept_)
print("Coefficients:\n" , model.coef_)

#Find the linear regression's model mean-squared-error on the training subset
predicted_weight_training = model.predict(fishmarket_num)
mse_training = mean_squared_error(fishmarket_label, predicted_weight_training)
print("Mean Squared Error on Training Set:", mse_training)
root_mse_train = np.sqrt(mse_training)
print("Root Mean Squared Error on Training Set:", root_mse_train)
print("")

#--NOW, WE USE THE ANALYTICAL APPROACH,i.e. Least Square Approximations--
X = strat_train_set.drop({'Species','Weight'}, axis = 1)
y = fishmarket_label

#Proceed with calculations
X_arr = X.to_numpy()
y_arr = y.to_numpy()

#Add column of 1's to first column of X_arr
#First, find out the number of rows
shape_X = X_arr.shape
ones_vec = np.ones((shape_X[0],1))
X_true = np.concatenate((ones_vec, X_arr), axis = 1)

Xt = X_true.transpose()
omega = (np.linalg.inv(Xt.dot(X_true))).dot(Xt.dot(y_arr))
print("Omega values using analytical method:\n", omega)


#Now we predict the entire training data set
y_predict = X_true.dot(omega.transpose())
mse_training_analytical = mean_squared_error(y_arr, y_predict)
print("Mean Squared Error on Training Set using analytical method:", mse_training_analytical)
root_mse_analytical = np.sqrt(mse_training_analytical)
print("Root Mean Squared Error on Training Set for analytical method:", root_mse_analytical)
print("\n")

#Testing using Test data set
fishmarket_test_num = strat_test_set.drop({'Species','Weight'}, axis = 1)
fishmarket_test_label = strat_test_set['Weight']

predicted_weight_test = model.predict(fishmarket_test_num)
mse_test_test = mean_squared_error(fishmarket_test_label, predicted_weight_test)
print("--TEST SET--")
print("Mean Squared Error on Test Set:", mse_test_test)
root_mse_test_test = np.sqrt(mse_test_test)
print("Root Mean Squared Error on Test Set:", root_mse_test_test)
r2_score = r2_score(fishmarket_test_label, predicted_weight_test)
print("R2 Score on Test Set:", r2_score)