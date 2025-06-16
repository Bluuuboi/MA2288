#Reference: https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
pd.set_option('display.max_colwidth', 200)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


dataset = fetch_20newsgroups(shuffle=True, random_state=42)
documents = dataset.data
# print(len(documents))
# print(dataset.target_names)

#Data Preprocessing
#Idea:
#Remove punctuations, numbers and special characters
#Then, remove shorter words, since they don't usually contain useful information
#Lastly, make all the text lowercase to nullify case sensitivity
news_df = pd.DataFrame({'document': documents})
# print(news_df)
#remove everything except alphabets
news_df['clean_doc'] = news_df['document'].str.replace('[^a-zA-Z]','')
#removing short words
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#make all text lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

#Good to remove stop words, since they are mostly clutter and hardly carry any information
# nltk.download('stopwords')
stop_words = stopwords.words('english')
#Tokenization
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
#remove stopwords
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
#detokenization
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc
# print(news_df['clean_doc'])

#Constructing a Document-Term Matrix, keep top 1000 terms
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.5, smooth_idf=True)
X = vectorizer.fit_transform(news_df['clean_doc'])
# print(X.shape)

#Topic Modelling, we will use Truncated SVD
#Since data comes from 20 different newsgroups, we'll try to have 20 topics for our text data
#SVD represents the documents and words in vectors
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=42)
svd_model.fit(X)
# print(len(svd_model.components_))

terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
    print('Topic ' + str(i) + ': ')
    for t in sorted_terms:
        print(t[0])
        print(' ')

#Topic Visualization
#Technique used here to visualise is called UMAP (Uniform Manifold Approximation and Projection)
import umap.umap_ as umap

X_topics = svd_model.fit_transform(X)
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state = 42).fit_transform(X_topics)

plt.figure(figsize=(10, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c = dataset.target, s = 10, edgecolor = 'none')
plt.show()
