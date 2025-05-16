import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)
og_data = pd.read_csv('Data_Sets/Salary Data.csv')
og_data = og_data.drop('Job Title', axis=1)
# print(og_data.head())
# print(og_data.describe())

og_data['Gender'] = og_data['Gender'].map({'Female': 0, 'Male': 1})
og_data['Education Level'] = og_data["Education Level"].map({"Bachelor's": 0, "Master's": 1, "PhD": 2})
pd.set_option('display.max_columns', None)
# print(og_data.head())

#Correlation heatmap
plt.figure(figsize = (10,10))
corr = og_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
# plt.tight_layout()
# plt.show()


og_data['Salary Range'] = pd.qcut(og_data['Salary'], q=4,labels=False)
salary_bins = pd.qcut(og_data['Salary'], q=4)
# print(og_data.head())
# print(salary_bins.cat.categories)

x_train,x_test,y_train,y_test = train_test_split(og_data, og_data['Salary'], test_size=0.2, random_state=30)

# print(x_train['Salary Range'].value_counts()/len(x_train))
# print(og_data['Salary Range'].value_counts()/len(og_data))

#Isolate numerical values and drop target feature(Drop 'Salary' and 'Salary Range' Columns)
train_num = x_train.drop({'Salary','Salary Range'}, axis=1)
train_num = train_num.dropna()
print(train_num.head())
#Isolate label feature
train_label = y_train.dropna()

model = LinearRegression()
model.fit(train_num, train_label)

#Print omegas
print("")
print("Omega values using scikit are:")
print(model.coef_)
print(model.score(train_num, train_label))

test_num = x_test.drop({'Salary','Salary Range'}, axis=1)
test_num = test_num.dropna()
test_label = y_test.dropna()

#Testing with test set
predicted_salary = model.predict(test_num)
mse_test = mean_squared_error(test_label, predicted_salary)
print("MSE FOR TEST SET:", mse_test)
r2_score = r2_score(test_label, predicted_salary)
print("R2 SCORE FOR TEST SET:", r2_score)

#Plotting predicted salary vs y_test
plt.figure(figsize = (10,10))
plt.scatter(y_test, predicted_salary, alpha=0.7, color='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
# plt.tight_layout()
# plt.show()






