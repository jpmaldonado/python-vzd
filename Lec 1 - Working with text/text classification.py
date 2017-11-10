# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:00:41 2017

@author: pc
"""

import pandas as pd

df = pd.read_csv("labeledTrainData.tsv", sep="\t")

# Get the first five columns
df.head()

# Other useful commands
df.info()

df.describe()

### Import vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words

sw = stop_words.ENGLISH_STOP_WORDS

# Look at the words
list(sw)[0:10]

count_vect = CountVectorizer(stop_words=sw, 
                             token_pattern=r'[a-z]{3,}')

X_counts = count_vect.fit_transform(df['review'])


# Word count
count_vect.vocabulary_.get("trashy")



# Using the classifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X_counts,y)


clf = MultinomialNB()

# Train
clf.fit(X_train,y_train)

# Generate predictions
y_preds = clf.predict(X_test)


# Analyzing model performance
from sklearn.metrics import log_loss
log_loss(y_test,y_preds)

# Mean squared error
sum((y_preds-y_test)**2)/len(y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_preds)

# How to look inside the black box?
feature_names = count_vect.get_feature_names()
coefs_with_feat_names = sorted(zip(clf.coef_[0],feature_names))
cat1 = coefs_with_feat_names[:10]
cat2 = coefs_with_feat_names[:-11:-1]

cat1
cat2


## How do I use this thing?
my_rev = ["I think this movie was really bad"]
x_my_rev = count_vect.transform(my_rev)
x_my_rev

feeling_of_my_rev = clf.predict(x_my_rev)
feeling_of_my_rev


















