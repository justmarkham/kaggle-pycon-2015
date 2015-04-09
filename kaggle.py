'''
Code from my first place submission in Kaggle's PyCon 2015 competition
https://github.com/justmarkham/kaggle-pycon-2015

Kevin Markham
kevin@dataschool.io
http://dataschool.io
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


## FILE READING AND FEATURE ENGINEERING

def make_features(filename):

    # read in a CSV file and define the first column as the index
    df = pd.read_csv(filename, index_col=0)
    
    # create feature that represents the length of the post title
    df['TitleLength'] = df.Title.apply(len)
    
    # create feature that represents the length of the body text
    df['BodyLength'] = df.BodyMarkdown.apply(len)
    
    # create feature that represents the number of tags
    df['has1'] = df.Tag1.notnull().astype(int)
    df['has2'] = df.Tag2.notnull().astype(int)
    df['has3'] = df.Tag3.notnull().astype(int)
    df['has4'] = df.Tag4.notnull().astype(int)
    df['has5'] = df.Tag5.notnull().astype(int)
    df['NumTags'] = df.has1 + df.has2 + df.has3 + df.has4 + df.has5
    
    # convert date fields from strings to datetime objects
    df['OwnerCreationDate'] = pd.to_datetime(df.OwnerCreationDate)
    df['PostCreationDate'] = pd.to_datetime(df.PostCreationDate)    
    
    # create feature that represents the age of the account (in days) at the time of posting
    df['OwnerAge'] = (df.PostCreationDate - df.OwnerCreationDate).dt.days

    # return a DataFrame
    return df

# add the same features to the training and testing data
train = make_features('train.csv')
test = make_features('test.csv')


## MODEL 1: Logistic Regression using 5 features

# create a list of features
cols = ['TitleLength', 'BodyLength', 'ReputationAtPostCreation', 'NumTags', 'OwnerAge']

# create X (feature matrix) and y (response vector)
X = train[cols]
y = train.OpenStatus

# instantiate model and fit with training data
lr = LogisticRegression()
lr.fit(X, y)

# calculate predicted probabilities on testing data
test_probs_lr = lr.predict_proba(test[cols])[:, 1]


## MODEL 2: Naive Bayes using vectorized post title as features

# instantiate vectorizer with default settings
vect = CountVectorizer()

# create document-term matrix from the training data
train_dtm = vect.fit_transform(train.Title)

# use vocabulary learned from training data to create document-term matrix from the testing data
test_dtm = vect.transform(test.Title)

# instantiate model and fit with training document-term matrix
nb = MultinomialNB()
nb.fit(train_dtm, train.OpenStatus)

# calculate predicted probabilities on the testing document-term matrix
test_probs_nb = nb.predict_proba(test_dtm)[:, 1]


## MODEL 3: Naive Bayes using vectorized body text as features

# instantiate vectorizer with optional arguments
vect = CountVectorizer(stop_words='english', max_features=20000)

# use the same pattern as model 2
train_dtm = vect.fit_transform(train.BodyMarkdown)
test_dtm = vect.transform(test.BodyMarkdown)
nb = MultinomialNB()
nb.fit(train_dtm, train.OpenStatus)
test_probs_nb2 = nb.predict_proba(test_dtm)[:, 1]


## ENSEMBLE THE MODELS AND SUBMIT

# calculate a weighted average of the predicted probabilities
test_probs = (test_probs_lr + test_probs_nb*4 + test_probs_nb2)/6

# create a DataFrame to store my submissions
sub = pd.DataFrame({'id':test.index, 'OpenStatus':test_probs}).set_index('id')

# write the submission to a CSV file
sub.to_csv('sub.csv')


## OTHER ASSORTED CODE (NOT USED IN BEST SUBMISSION)

# validate locally using 5-fold cross-validation
from sklearn.cross_validation import cross_val_score
cross_val_score(lr, X, y, cv=5, scoring='log_loss').mean()

# Random Forest model
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features='auto')
rfclf.fit(X, y)
test_probs_rf = rfclf.predict_proba(test[cols])[:, 1]
