{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9822150363783346\n",
      "0.8832391713747646\n",
      "col_0         0         1         2\n",
      "row_0                              \n",
      "0      0.827869  0.049180  0.122951\n",
      "1      0.014085  0.910798  0.075117\n",
      "2      0.081633  0.030612  0.887755\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "import string\n",
    "from nltk.corpus import stopwords\n",
    "import spacy\n",
    "from spacy.lang.lt.stop_words import STOP_WORDS\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from PIL import Image\n",
    "from wordcloud import WordCloud\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import numpy as np\n",
    "import itertools\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def text_processing(text):\n",
    "     # 1. Removal of Punctuation Marks\n",
    "    nopunct=[char for char in text if char not in string.punctuation]\n",
    "    nopunct=''.join(nopunct)\n",
    "    #Lemmatising\n",
    "    doc = nlp(nopunct)\n",
    "    string1=''\n",
    "    for word in doc:\n",
    "        string1+=word.lemma_+ ' '\n",
    "    string1=string1[:-1]\n",
    "    # 3. Removal of Stopwords\n",
    "    string1 = nlp(string1)\n",
    "    return [word.text.lower() for word in string1 if not word.is_stop]\n",
    "\n",
    "def label_encoding(data):\n",
    "    y = data['Author']\n",
    "    labelencoder = LabelEncoder()\n",
    "    y = labelencoder.fit_transform(y)\n",
    "    return y\n",
    "\n",
    "def word_cloud_visualisation(data,X):\n",
    "    wordcloud1 = WordCloud().generate(X[1]) # Biliunas\n",
    "    wordcloud2 = WordCloud().generate(X[900]) # Zemaite\n",
    "    wordcloud3 = WordCloud().generate(X[301]) # Donelaitis\n",
    "     \n",
    "    print(X[1])\n",
    "    print(data['Author'][1])\n",
    "    plt.imshow(wordcloud1, interpolation='bilinear')\n",
    "    plt.show()\n",
    "    print(X[900])\n",
    "    print(data['Author'][900])\n",
    "    plt.imshow(wordcloud2, interpolation='bilinear')\n",
    "    plt.show()\n",
    "    print(X[301])\n",
    "    print(data['Author'][301])\n",
    "    plt.imshow(wordcloud3, interpolation='bilinear')\n",
    "    plt.show()\n",
    "    \n",
    "def model_trainning(text_bow_train, y_train):\n",
    "    model = MultinomialNB()\n",
    "    model = model.fit(text_bow_train, y_train)\n",
    "    return model\n",
    "\n",
    "def show_confusion_matrix(y_test, predictions):\n",
    "    matrix = pd.crosstab(y_test,predictions)\n",
    "    #matrix = confusion_matrix(y_test,predictions)\n",
    "    norm_matrix = matrix/matrix.sum(axis = 1)[:,np.newaxis]\n",
    "    print (norm_matrix)\n",
    "    plt.matshow(norm_matrix, cmap=plt.cm.Reds)\n",
    "    plt.colorbar()\n",
    "    plt.ylabel(\"Actual\")\n",
    "    plt.xlabel(\"Predicted\")\n",
    "    \n",
    "data = pd.read_csv('train1.csv')\n",
    "nlp = spacy.load(\"lt_core_news_sm\")\n",
    "spacy_stopwords = spacy.lang.lt.stop_words.STOP_WORDS\n",
    "\n",
    "X=data['Text']\n",
    "y = label_encoding(data)\n",
    "#word_cloud_visualisation(data,X)\n",
    "#Splitting data into testing and training data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)\n",
    "bow_transformer=CountVectorizer(analyzer=text_processing)\n",
    "text_bow_train=bow_transformer.fit_transform(X_train)\n",
    "#print(bow_transformer.get_feature_names())\n",
    "#print(text_bow_train.toarray())\n",
    "text_bow_test=bow_transformer.transform(X_test)\n",
    "\n",
    "model = model_trainning(text_bow_train,y_train)\n",
    "print (model.score(text_bow_train, y_train))\n",
    "print (model.score(text_bow_test, y_test))\n",
    "\n",
    "predictions = model.predict(text_bow_test)\n",
    "show_confusion_matrix(y_test,predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
