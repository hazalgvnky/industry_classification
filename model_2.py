#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[2]:


col_data = {
    'text' : str,
    'industry' : str, 
}
cols=['text','industry']
df = pd.read_csv('/Users/macbookretina/Desktop/user_classification_clean_data.csv',header= None,dtype=col_data, parse_dates=True, names= list(col_data.keys()))


# In[3]:


df = df.sort_values('industry')
df = df.dropna()


# In[4]:


df['in_id'] = df['industry'].factorize()[0]
category_id_df = df[['industry', 'in_id']].drop_duplicates().sort_values('in_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['in_id', 'industry']].values)


# In[5]:


df = df[df["in_id"] != 48]


# In[6]:


df


# In[7]:


# TEST CASE 
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['industry'], random_state = 0)
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[8]:


from sklearn import metrics
#clf.predict(count_vect.transform([testText]))
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


# In[ ]:


clf = RandomForestClassifier().fit(X_train_tfidf, y_train)


# In[ ]:


# classification report RandomForestClassifier
y_pred = clf.predict(X_test_tfidf)
print(metrics.classification_report(y_test, y_pred))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


mlf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[ ]:


# classification report MultinomialNB
y_pred = mlf.predict(X_test_tfidf)
print(metrics.classification_report(y_test, y_pred))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


xlf = XGBClassifier().fit(X_train_tfidf, y_train)


# In[ ]:


# classification report MultinomialNB
y_pred = xlf.predict(X_test_tfidf)
print(metrics.classification_report(y_test, y_pred))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

