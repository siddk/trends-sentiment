"""
sentiment_analysis.py

Create a naive bayes classifier to take in a phrase (set of words), and
sort into one of two categories: pos(itive), and neg(ative).

Compared to the twitter trends project, this is more accurate in the sense
that this is built on an actual statistical model, rather than a dictionary
of sentiment values.

Built using Python's NLTK library, with the Naive Bayes Classifier trained on
a set of movie_reviews, included in the NLTK corpora. For sentiment analysis,
movie reviews and twitter tweets don't differ all too much, so the same classifier
defined here works for tweets as well.
"""
__author__ = "Sidd Karamcheti"
__author__ = "Ulysse Carion"

import nltk.classify.util
import pickle
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Only the `num_words_considered` most informative words are chosen to be
# considered when finding a phrase's sentiment value.
#
# Higher numbers means you take into account more words, but also means that
# some of the noisier, less informative, words are going to be included in your
# conclusions.
num_words_considered = 10000

# What percent of the data will we use for training? The rest will be used for
# evaluating the accuracy of our model.
proportion_training = 0.75

neg_review_ids = movie_reviews.fileids('neg')
pos_review_ids = movie_reviews.fileids('pos')

neg_reviews = [movie_reviews.words(fileids = [f]) for f in neg_review_ids]
pos_reviews = [movie_reviews.words(fileids = [f]) for f in pos_review_ids]


def train_classifier(feature_fn):
    """
    Given a `feature_fn` that converts a sentence into a set of its features,
    generates a classifier.

    Returns the classifier, and an estimate of the classifier's accuracy.
    """

    neg_features = [(feature_fn(review), 'neg') for review in neg_reviews]
    pos_features = [(feature_fn(review), 'pos') for review in pos_reviews]

    # The index at which we split the features, where things before this index
    # are training data, and the rest is testing data.
    neg_cutoff = int(len(neg_features) * proportion_training)
    pos_cutoff = int(len(pos_features) * proportion_training)

    train_features = neg_features[:neg_cutoff] + pos_features[:pos_cutoff]
    test_features  = neg_features[neg_cutoff:] + pos_features[pos_cutoff:]

    classifier = NaiveBayesClassifier.train(train_features)

    return classifier, nltk.classify.util.accuracy(classifier, test_features)

# A frequency distribution of how often words appear.
word_freq_dist = FreqDist()

# A frequency distribution of how often words are associated with 'pos' or
# 'neg'.
label_word_freq_dist = ConditionalFreqDist()

# Build up the frequency distributions.
for label in ['pos', 'neg']:
    for word in movie_reviews.words(categories = [label]):
        word_freq_dist[word.lower()] += 1
        label_word_freq_dist[label][word.lower()] += 1

num_pos_words = label_word_freq_dist['pos'].N()
num_neg_words = label_word_freq_dist['neg'].N()
total_num_words = num_pos_words + num_neg_words

def word_information_score(word, freq):
    pos_score = BigramAssocMeasures.chi_sq(
        label_word_freq_dist['pos'][word],
        (freq, num_pos_words),
        total_num_words)

    neg_score = BigramAssocMeasures.chi_sq(
        label_word_freq_dist['neg'][word],
        (freq, num_neg_words),
        total_num_words)

    return pos_score + neg_score

word_info_scores = {word: word_information_score(word, freq)
    for word, freq in word_freq_dist.items()}

sorted_by_score = sorted(word_info_scores.items(),
    key = lambda word_score_pair: word_score_pair[1],
    reverse = True)[:num_words_considered]

best_words = set([word for word, score in sorted_by_score])

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

classifier, accuracy = train_classifier(best_word_features)

# Save classifier in separate file - it only needs to be trained once
with open('naive_bayes.pickle', 'wb') as f
    pickle.dump(classifier, f)
