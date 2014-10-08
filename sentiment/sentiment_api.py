"""
sentiment_api.py

Provides an abstraction for accessing our Naive Bayes classifier, one that
is simple enough to use for potential iterations of the twitter trends project, but
still complex enough to be extended.

For the time being, we only categorize phrases into two categories, "pos", and "neg",
but we provide an extra function to return an integer value (1 or -1), for better integration
into the trends project.

POSSIBLE EXTENSIONS: This is a bare bones implementation of a classifier built in NLTK, but
a possible extension is to use part-of-speech tagging (another feature of NLTK) to separate
tweets into sets of different mini-phrases (i.e. noun-adjective phrases), to be fed into our
phrase classifier.

This would paint a better picture of the sentiment of the tweet as a whole, and would eliminate
a lot of the shortcomings of the current project's entire sentiment-by-word idea, which can often
be inaccurate ("not", "good" as individual sentiment values, whereas "not good" produces a single,
accurate value).
"""

__author__ = "Sidd Karamcheti"
__author__ = "Ulysse Carion"

import pickle

# Initialize classifier
classifier = ""
with open('naive_bayes.pickle') as f:
    classifier = pickle.load(f)


# Given a phrase, what is the sentiment of that phrase?
#
# If a phrase is deemed to be positive, the return value will be 'pos', or 'neg'
# otherwise.
def phrase_sentiment(phrase):
    x = best_word_features(phrase.split())
    print(classifier.classify(x))