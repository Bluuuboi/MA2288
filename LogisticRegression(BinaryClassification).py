import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#Binary classification, so we will only classify an image as either that of an 8 or not an 8
mnist_data = fetch_openml('mnist_784', version=1, data_home='Data_Sets/MNIST')
# print(mnist_data.keys())
# print(mnist_data.url)

X,y_true = mnist_data['data'],mnist_data['target']
print('Data without target label has shape:', X.shape)
print('Type of data in each feature cell:', type(X.iloc[1,1]))
print('Target label have shape:', y_true.shape)
print('Type of data in each label cell:', type(y_true.iloc[1,]))

# print("")
# print(type(X))
# print(type(y_true))

#Change the labels from string to float
y_true = y_true.astype(np.uint8)
print('\nType of data in each label cell:', type(y_true.iloc[1,]))

#Training and Test subsets
#Standard scaler to improve time for convergence
#If output/prediction is a classification, typically no need to reverse the scaling, but if it's a value, should reverse the scaling.
scaler = StandardScaler()
X_train,X_test,y_true_train,y_true_test= X.iloc[0:60000,:],X.iloc[60000:70000,:],y_true.iloc[0:60000],y_true.iloc[60000:70000]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#For this binary classifier, our target is only 8
y_true_train_8, y_true_test_8=(y_true_train==8), (y_true_test==8)

start = time.time()
clf_lr_model = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000)
clf_lr_model.fit(X_train_scaled,y_true_train_8)

#Model accuracy on the whole training subset
score = clf_lr_model.score(X_train_scaled,y_true_train_8)
print(score)
end = time.time()
print('Time taken to train model:', end-start)

#Performance measures
start = time.time()
y_train_pred = cross_val_predict(clf_lr_model, X_train_scaled,y_true_train_8, cv=3)
end = time.time()
print('Time taken to train model:', end-start)

#Confusion Matrix
#Row 0: True Negative | False Positive
#Row 1: False Negative| True Positive
conf_matrix = confusion_matrix(y_true_train_8, y_train_pred)
print('Confusion matrix:\n', conf_matrix)
#Normalise confusion matrix
row_totals = conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_normalized = conf_matrix/row_totals
print('Normalized confusion matrix:\n', conf_matrix_normalized)

#For this colorscale, the lighter the box is, the higher the proportion of correctly predicted negatives/positives
plt.matshow(conf_matrix_normalized,cmap='gray')
plt.savefig('Fig_confusion_matrix_binary')
plt.show()

print('\n Precision score=',precision_score(y_true_train_8,y_train_pred))

print('\n Recall score=',recall_score(y_true_train_8,y_train_pred))

print('\n F1 score=',f1_score(y_true_train_8,y_train_pred))

