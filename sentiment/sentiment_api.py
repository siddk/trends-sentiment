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

from nltk.classify import NaiveBayesClassifier
from sentiment_classifier import best_word_features
from string import ascii_letters
import pickle

with open('naive_bayes.pickle', 'rb') as f:
    classifier = pickle.load(f)

def extract_words(tweet_text):
    """
    Return the words in a tweet, not including punctuation.

    >>> extract_words('anything else.....not my job')
    ['anything', 'else', 'not', 'my', 'job']
    >>> extract_words('i love my job. #winning')
    ['i', 'love', 'my', 'job', 'winning']
    >>> extract_words(('make justin # 1 by tweeting #vma #justinbieber :)'))
    ['make', 'justin', 'by', 'tweeting', 'vma', 'justinbieber']
    >>> extract_words("paperclips! they're so awesome, cool, & useful!")
    ['paperclips', 'they', 're', 'so', 'awesome', 'cool', 'useful']
    >>> extract_words('@(cat$.on^#$my&@keyboard***@#*')
    ['cat', 'on', 'my', 'keyboard']
    """
    filtered_string = ''.join([l if l in ascii_letters else ' ' for l in tweet_text])
    return filtered_string.split()

def phrase_sentiment_string(phrase):
    """
    Use the Naive Bayes classifier to characterize the sentiment of
    a given phrase. Uses function extract_words from the trends project
    to break up a phrase.

    Returns a string 'pos', or 'neg', depending on the sentiment of the phrase.

    >>> phrase_sentiment_string("I love my mom")
    'pos'
    >>> phrase_sentiment_string("I hate my mom")
    'neg'
    """
    return classifier.classify(best_word_features(extract_words(phrase)))

def phrase_sentiment_value(phrase):
    """
    Characterize the sentiment of a phrase as a single value.

    Returns either 1 or -1 depending on the return value from phrase_sentiment_string.
    1 if there is a positive sentiment, -1 if there is a negative sentiment.

    >>> phrase_sentiment_value("I love my mom")
    1
    >>> phrase_sentiment_value("I hate my mom")
    -1
    """
    return 1 if phrase_sentiment_string(phrase) == 'pos' else -1

def overall_tweet_sentiment(tweet):
    """
    Given a tweet, characterize the overall sentiment as a value between 1 and -1, using
    part-of-speech tagging. Form relationships between the words of a tweet
    (i.e. "not", "good" -> "not good"), and use the sentiment values of each "mini-phrase"
    to calculate an average sentiment value for the tweet as a whole.

    Return a number between -1 and 1, depending on the sentiment of a tweet as a whole.
    """
    pass
