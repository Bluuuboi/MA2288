import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("../Data_Sets/Raisin_Dataset.csv")
df['Class'] = df['Class'].map({'Kecimen':0,'Besni':1})
# print(df.head())

plt.figure(figsize = (10,10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
# plt.show()

x_train,x_test,y_train,y_test = train_test_split(df,df['Class'],test_size=0.2,random_state=42)
# print(x_train)
# print(type(x_train))
# print(y_train)

#Drop label features
x_train = x_train.drop('Class',axis=1)
x_test = x_test.drop('Class',axis=1)

#Create a SVM Classifier, rbf just means Gaussian
clf = svm.SVC(kernel='rbf')
#Train the model using training sets
clf.fit(x_train,y_train)

#Predict using test set
y_pred = clf.predict(x_test)

#Evaluate Model
accuracy = metrics.accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)

#w's can only be obtained if we use linear kernel.
#Kernel trick is utilised for the rest, i.e. poly and Gaussian Kernels, where input data is mapped to some higher dimensional space.

