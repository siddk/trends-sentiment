import nltk.classify.util
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

# Given a `feature_fn` that converts a sentence into a set of its features,
# generates a classifier.
#
# Returns the classifier, and an estimate of the classifier's accuracy.
def train_classifier(feature_fn):
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

# Given a phrase, what is the sentiment of that phrase?
#
# If a phrase is deemed to be positive, the return value will be 'pos', or 'neg'
# otherwise.
def phrase_sentiment(phrase):
    return classifier.classify(best_word_features(phrase.split()))
