#Sentiment Analysis#

This was a project inspired by the [twitter-trends](http://inst.eecs.berkeley.edu/~cs61a/fa14/proj/trends/) project in our class CS 61A at the University of California - Berkeley, Fall 2014, under Professor John DeNero. In the course of completing the project, we noticed that finding sentiments for given words was done in a very inefficient manner - essentially looking up pre-defined values in a gigantic dictionary.

To get around this, we used Python's [NLTK](http://www.nltk.org/) - Natural Language Toolkit to create a Naive Bayesian Classifier, and a surrounding abstraction. Details of our classifier and abstraction are below, and the necessary files are located in the sentiment folder above.

In order for things to run smoothly, it is important that you have the following installed:

1. Python3
2. NLTK, and all its dependencies
3. NLTK Data, which contain all the training corpora

You may also run into an OS Error while running the train function for the Naive Bayes classifier, in which case you should increase your open file limit to > 2000.

###sentiment_classifier.py###

We create a naive bayes classifier to take in a phrase (set of words), and
sort into one of two categories: pos(itive), and neg(ative).

Compared to the twitter trends project, this is more accurate in the sense
that this is built on an actual statistical model, rather than a dictionary
of sentiment values.

Built using Python's NLTK library, with the Naive Bayes Classifier trained on
a set of movie_reviews, included in the NLTK corpora. For sentiment analysis,
movie reviews and twitter tweets don't differ all too much, so the same classifier defined here works for tweets as well.

###sentiment_api.py###

Provides an abstraction for accessing our Naive Bayes classifier, one that
is simple enough to use for potential iterations of the twitter trends project, but still complex enough to be extended.

For the time being, we only categorize phrases into two categories, "pos", and "neg", but we provide an extra function to return an integer value (1 or -1), for better integration into other projects. The Israel-Palestine parent project will rely heavily on this functionality.

Documentation of our phrase_sentiment_string function:

```python
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
```

---
For further information, consult the documentation in each of the above files, and feel free to open issues if there are any problems in using/running the above classifier.
