import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

breast = load_breast_cancer()
breast_data = breast.data
# print(breast_data.shape)

breast_labels = breast.target
# print(breast_labels.shape)

labels = np.reshape(breast_labels, (569,1))
final_breast_data = np.concatenate([breast_data, labels], axis=1)
# print(final_breast_data.shape)

breast_dataset = pd.DataFrame(final_breast_data)
# print(breast_dataset.head())
features = breast.feature_names
# print(features)

features_labels= np.append(features, 'label')
breast_dataset.columns = features_labels
# print(breast_dataset.head())

breast_dataset.replace({'label':{0:'Benign', 1:'Malignant'}}, inplace=True)
# print(breast_dataset.head())
# print(breast_dataset.tail())

scaler = StandardScaler()
x = breast_dataset.loc[:, features].values
x = scaler.fit_transform(x)
# print(np.mean(x))
# print(np.std(x))

#Convert the normalised features into a tabular format using DataFrame
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
#x.shape[1] = 30
print(normalised_breast.head())

#PCA
#For this case, we choose to have two principal components
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)
#Now we create a DataFrame for the above
principal_breast_df = pd.DataFrame(data=principalComponents_breast, columns=['principal component 1', 'principal component 2'])
print(principal_breast_df.head())

print('Explained variability per principal component: {}'.format(pca_breast.explained_variance_ratio_))
#I.e. about 36.8% of data was lost while projecting 30-dim data to 2-dim data

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_df.loc[indicesToKeep, 'principal component 1']
                , principal_breast_df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
plt.legend(targets,prop={'size': 15})
plt.show()













