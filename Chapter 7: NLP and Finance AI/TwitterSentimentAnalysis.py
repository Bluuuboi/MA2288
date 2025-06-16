#Reference: https://www.hackersrealm.net/post/twitter-sentiment-analysis-using-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.stem.porter import PorterStemmer

df = pd.read_csv("../Data_Sets/Twitter Sentiments.csv")
# print(df.head())

#Label: 1 if it's a negative sentiment, 0 otherwise

#Preprocessing
#Function to remove patterns in the input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

#remove twitter handles (i.e. @user)
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], r"@[\w]*")
# print(df.head())
#remove special characters, numbers and punctuations
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]"," ")
# print(df.head())
#remove short words
df['clean_tweet'] =  df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
# print(df.head())

#Individual words considered as tokens
tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
# print(tokenized_tweet.head())

#Stem the words (i.e. simplifying the words to their most basic form)
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
# print(tokenized_tweet.head())

#Combine words into a single sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
df['clean_tweet'] = tokenized_tweet
# print(df.head())


#Exploratory Data Analysis (EDA)
#visualize the frequent words
from wordcloud import WordCloud
all_words = " ".join(sentence for sentence in df['clean_tweet'])
wordcloud = WordCloud(width=800, height=500, random_state=4).generate(all_words)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis("off")


#visualization for frequent words with positive sentiment
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==0]])
wordcloud = WordCloud(width=800, height=500, random_state=4).generate(all_words)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")


#visualization for frequent words with negative sentiment
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==1]])
wordcloud = WordCloud(width=800, height=500, random_state=4).generate(all_words)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.show()


#Function to extract the hashtags
def hashtag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#[\w]+", tweet)
        hashtags.append(ht)
    return hashtags
#extract hashtags from positive tweets
ht_positive = hashtag_extract(df['clean_tweet'][df['label']==0])
#extract hashtags from negative tweets
ht_negative = hashtag_extract(df['clean_tweet'][df['label']==1])
#printing the first five for each
# print(ht_positive[:5])
# print(ht_negative[:5])
#unnest list for easier viewing
ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, [])
# print(ht_positive[:5])
# print(ht_negative[:5])


#Convert dictionary into DataFrame to list hashtags with word count
#For positive tweets
freq = nltk.FreqDist(ht_positive)
positive_words = pd.DataFrame({'Hashtag': list(freq.keys()), 'Count': list(freq.values())})
print(positive_words.head())
#For negative tweets
freq = nltk.FreqDist(ht_negative)
negative_words = pd.DataFrame({'Hashtag': list(freq.keys()), 'Count': list(freq.values())})
print(negative_words.head())


#Visualizing top ten hashtags with high frequency
positive_words = positive_words.nlargest(10, 'Count')
plt.figure(figsize=(10, 10))
sns.barplot(data=positive_words, x='Hashtag', y='Count')

negative_words = negative_words.nlargest(10, 'Count')
plt.figure(figsize=(10, 10))
sns.barplot(data=negative_words, x='Hashtag', y='Count')
# plt.show()


#Input split, which is a preprocessing step for feature selection/feature extraction of words,
#in order to convert them into vectors for machine to understand
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#Extraction of data into vectors for training and testing
bow_vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])

#Splitting the data, test size = 25%
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], test_size=0.25, random_state=42)


#Model training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

#training
model = LogisticRegression()
model.fit(x_train, y_train)

#testing
pred = model.predict(x_test)
f1_score = f1_score(y_test, pred)
accuracy_score = accuracy_score(y_test, pred)
print(f1_score)
print(accuracy_score)
