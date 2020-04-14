#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re
import string
from nltk.corpus import stopwords
import spacy
from spacy.lang.lt.stop_words import STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt

def text_processing(text):
     # 1. Removal of Punctuation Marks
    nopunct=[char for char in text if char not in string.punctuation]
    nopunct=''.join(nopunct)
    #Lemmatising
    doc = nlp(nopunct)
    string1=''
    for word in doc:
        string1+=word.lemma_+ ' '
    string1=string1[:-1]
    # 3. Removal of Stopwords
    string1 = nlp(string1)
    return [word.text.lower() for word in string1 if not word.is_stop]

def label_encoding(data):
    y = data['Author']
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    return y

def word_cloud_visualisation(data,X):
    wordcloud1 = WordCloud().generate(X[1]) # Biliunas
    wordcloud2 = WordCloud().generate(X[900]) # Zemaite
    wordcloud3 = WordCloud().generate(X[301]) # Donelaitis
     
    print(X[1])
    print(data['Author'][1])
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.show()
    print(X[900])
    print(data['Author'][900])
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.show()
    print(X[301])
    print(data['Author'][301])
    plt.imshow(wordcloud3, interpolation='bilinear')
    plt.show()
    
def model_trainning(text_bow_train, y_train):
    model = MultinomialNB()
    model = model.fit(text_bow_train, y_train)
    return model

def show_confusion_matrix(y_test, predictions):
    matrix = pd.crosstab(y_test,predictions)
    #matrix = confusion_matrix(y_test,predictions)
    norm_matrix = matrix/matrix.sum(axis = 1)[:,np.newaxis]
    print (norm_matrix)
    plt.matshow(norm_matrix, cmap=plt.cm.Reds)
    plt.colorbar()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    
data = pd.read_csv('train1.csv')
nlp = spacy.load("lt_core_news_sm")
spacy_stopwords = spacy.lang.lt.stop_words.STOP_WORDS

X=data['Text']
y = label_encoding(data)
#word_cloud_visualisation(data,X)
#Splitting data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
bow_transformer=CountVectorizer(analyzer=text_processing)
text_bow_train=bow_transformer.fit_transform(X_train)
#print(bow_transformer.get_feature_names())
#print(text_bow_train.toarray())
text_bow_test=bow_transformer.transform(X_test)

model = model_trainning(text_bow_train,y_train)
print (model.score(text_bow_train, y_train))
print (model.score(text_bow_test, y_test))

predictions = model.predict(text_bow_test)
show_confusion_matrix(y_test,predictions)

