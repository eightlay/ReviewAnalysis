# Import packages
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Import source code
path = os.getcwd()
path = path[:path.rfind('\\') + 1]
sys.path.insert(1, path + 'src')
from choose_model import *
from preprocessing import *

# Create initial dataset and prepare it. 
# See "Credits" section in README of this project for references.
'''
data = ReviewSentimentMatch(path + r'data\origin\\')
data.to_csv(path + r'data\kinopoisk3000.csv')

# Prepare dataset
data = PrepareData(data, language='russian')
data.to_csv(path + r'data\prepared_kinopoisk3000.csv')
'''

# Load dataset
data = pd.read_csv(path + r'data\prepared_kinopoisk3000.csv')

# Vectorize data
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(data.review).toarray()
y = np.array(data.sentiment.values)

# Split data
trainx, testx, trainy, testy = train_test_split(
    X, y, test_size=0.10, random_state=9)

models = [
    GaussianNB(),
    MultinomialNB(alpha=1.0, fit_prior=True),
    BernoulliNB(alpha=1.0, fit_prior=True)
]

# Choose the best model
best, score = ChooseBestModel(models, train_data=(
    trainx, trainy), test_data=(testx, testy))
print('Using prediction model with accuracy = {}%.'.format(np.round(score, 4) * 100))

# Feed testing reviews
decoder = {-1: 'bad', 0: 'neutral', 1: 'good'}

for review_path in ['test0.txt', 'test1.txt', 'test2.txt']:
    # Read review
    with open(review_path, "r", encoding='utf-8') as review:
        test_review = review.read()

    # Prepare review
    test_review_to_feed = FeedReview(test_review, 'russian')
    test_review_to_feed = cv.transform([test_review_to_feed]).toarray()

    # Predict estimation
    prediction = best.predict(test_review_to_feed)
    prediction = decoder[prediction[0]]

    # Print it
    print('    Review in {} rates the film as {} one.'.format(
        review_path, prediction))
