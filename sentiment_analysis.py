import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

# What percent of the data will we use for training? The rest will be used for
# evaluating the accuracy of our model.
proportion_training = 0.75

neg_review_ids = movie_reviews.fileids('neg')
pos_review_ids = movie_reviews.fileids('pos')

neg_words = [movie_reviews.words(fileids = [f]) for f in neg_review_ids]
pos_words = [movie_reviews.words(fileids = [f]) for f in pos_review_ids]

# Given a `feature_fn` that converts a sentence into a set of its features,
# generates a classifier.
#
# Returns the classifier, and an estimate of the classifier's accuracy.
def train_classifier(feature_fn):
    neg_features = [(feature_fn(neg_word), 'neg') for neg_word in neg_words]
    pos_features = [(feature_fn(pos_word), 'pos') for pos_word in pos_words]

    # The index at which we split the features, where things before this index
    # are training data, and the rest is testing data.
    neg_cutoff = int(len(neg_features) * proportion_training)
    pos_cutoff = int(len(pos_features) * proportion_training)

    train_features = neg_features[:neg_cutoff] + pos_features[:pos_cutoff]
    test_features  = neg_features[neg_cutoff:] + pos_features[pos_cutoff:]

    classifier = NaiveBayesClassifier.train(train_features)

    return classifier, nltk.classify.util.accuracy(classifier, test_features)

def word_feats(words):
    return dict([(word, True) for word in words])

classifier, accuracy = train_classifier(word_feats)

print('Estimated accuracy: {}'.format(accuracy))
classifier.show_most_informative_features()
