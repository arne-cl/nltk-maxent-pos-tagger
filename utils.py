# -*- coding: utf-8 -*-
# functions that aren't necessary, but useful when working w/ nltk-mxpost.py

from __future__ import division # always use "real" division (3 / 2 is 1.5 not 1)
import nltk, re, pprint
import cPickle as pickle
import time
import numpy

"""
General usage examples
----------------------

Using the BracketParseCorpusReader (see below) with files instead of
dictionaries is painfully SLOW. Therefore we have converted those files
into lists/dictiories and stored them in pickles/shelves for convenient reuse.

Files:
wsj_tagged_sents.pickle: wsj_tagged_sents (list) contains all 49208 tagged
    sentences from the WSJ corpus. generated with generate_tagged_sents_list()
wsj_tagged_sents.shelve: dito, but as a shelve (less memory consumption)
wsj_word_freqdist.pickle: wsj_word_freqdist is a FrequencyDistribution object
    containing 49817 words and their frequencies in the WSJ corpus
wsj_tag_freqdist.pickle: wsj_tag_freqdist is a FreqDist object containing all
    tags from the WSJ corpus


loading pickle files:
object_name = load_from_pickle(object_name, file_name)

"""

def generate_tagged_sents_list():
    """generate a list of all tagged sentence that we can store with Pickle / use from memory
       input: none, output: wsj_tagged_sents (list)."""
    wsj_tagged_sents = []
    for tagged_sent in ptb.tagged_sents():
        wsj_tagged_sents.append(tagged_sent)
    return wsj_tagged_sents

def generate_tagged_words_list():
    """generate a list of all tagged words. each list item is a tuple of the
    form (word, tag)"""
    wsj_tagged_words = []
    for tagged_word in ptb.tagged_words():
        wsj_tagged_words.append(tagged_word)
    return wsj_tagged_words


def generate_freqdists():
    """generate frequency distributions for words and tags with BracketParseCorpusReader's tagged_words() method.

    input: none, output: word_fredist, tag_freqdist."""
    word_freqdist = nltk.FreqDist(word for (word, tag) in ptb.tagged_words())
    tag_freqdist = nltk.FreqDist(tag for (word, tag) in ptb.tagged_words())
    return word_freqdist, tag_freqdist

def generate_freqdists_from_tagged_sents(tagged_sents):
    """generate frequency distributions from a list of tagged sents.

       input: tagged sentences (list). output: freqdist of words

       as nested list comprehension:

       word_freqdist = nltk.FreqDist(word
                                for tagged_sent in tagged_sents
                                for (word, tag) in tagged_sent)
    """
    word_freqdist = nltk.FreqDist()
    for tagged_sent in tagged_sents:
        for (word, tag) in tagged_sent:
            word_freqdist.inc(word)
    return word_freqdist

def save_to_pickle(object_name, file_name):
    """saves an object to a pickle file
    input: object name, pickle file name (as STRING)"""
    pickle_file = open(file_name, "w")
    pickle.dump(object_name, pickle_file)
    pickle_file.close()

def load_from_pickle(object_name, file_name):
    """loads an object from a pickle file into memory
       input: object name as STRING (as stored in pickle file), pickle file name (as STRING).
       output: object.
    """
    pickle_file = open(file_name, "r")
    object_name = pickle.load(pickle_file)
    pickle_file.close()
    return object_name

