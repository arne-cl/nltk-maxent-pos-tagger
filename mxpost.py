# -*- coding: utf-8 -*-
# Maximum Entropy Part-of-Speech Tagger for NLTK (Natural Language Toolkit)
# Author: Arne Neumann
# Licence: GPL 3

#__docformat__ = 'epytext en'

"""
A I{part-of-speech tagger} that uses NLTK's build-in L{Maximum Entropy
models<nltk.MaxentClassifier>} to find the most likely I{part-of-speech
tag} (POS) for each word in a given sequence.

The tagger will be trained on a corpus of tagged sentences. For every word
in the corpus, a C{tuple} consisting of a C{dictionary} of features from
the word's context (e.g. preceding/succeeding words and tags, word
prefixes/suffixes etc.) and the word's tag will be generated.
The maximum entropy classifier will learn a model from these tuples that
will be used by the tagger to find the most likely POS-tag for any given
word, even unseen ones.

The tagger and the featuresets chosen for training are implemented as described
in Ratnaparkhi, Adwait (1996). A Maximum Entropy Model for Part-Of-Speech
Tagging. In Proceedings of the ARPA Human Language Technology Workshop. Pages
250-255.

Usage notes:
============

Please install the MEGAM package (http://hal3.name/megam),
otherwise training will take forever.

To use the demo, please install either 'brown' or 'treebank' with::

    import nltk
    nltk.download()

in the Python interpreter. Proper usage of demo() and all other functions and
methods is described below.
"""

import time
import re
from collections import defaultdict

from nltk import TaggerI, FreqDist, untag, config_megam
from nltk.classify.maxent import MaxentClassifier
                  

PATH_TO_MEGAM_EXECUTABLE = "/usr/bin/megam"
config_megam(PATH_TO_MEGAM_EXECUTABLE)


class MaxentPosTagger(TaggerI):
    """
    MaxentPosTagger is a part-of-speech tagger based on Maximum Entropy models.
    """
    def train(self, train_sents, algorithm='megam', rare_word_cutoff=5,
              rare_feat_cutoff=5, uppercase_letters='[A-Z]', trace=3,
              **cutoffs):
        """
        MaxentPosTagger trains a Maximum Entropy model from a C{list} of tagged
        sentences.

        @type train_sents: C{list} of C{list} of tuples of (C{str}, C{str})
        @param train_sents: A list of tagged sentences. Each sentence is
        represented by a list of tuples. Each tuple holds two strings, a
        word and its tag, e.g. ('company','NN').

        @type algorithm: C{str}
        @param algorithm: The algorithm that is used by
        L{nltk.MaxentClassifier.train()} to train and optimise the model. It is
        B{strongly recommended} to use the C{LM-BFGS} algorithm provided by the
        external package U{megam<http://hal3.name/megam/>} as it is much faster
        and uses less memory than any of the algorithms provided by NLTK (i.e.
        C{GIS}, C{IIS}) or L{scipy} (e.g. C{CG} and C{BFGS}).

        @type rare_word_cutoff: C{int}
        @param rare_word_cutoff: Words with less occurrences than
        C{rare_word_cutoff} will be treated differently by L{extract_feats}
        than non-rare words (cf. Ratnaparkhi 1996).

        @type rare_feat_cutoff: C{int}
        @param rare_feat_cutoff: ignore features that occur less than
        C{rare_feat_cutoff} during training.

        @type uppercase_letters: C{regex}
        @param uppercase_letters: a regular expression that covers all
        uppercase letters of the language of your corpus (e.g. '[A-ZÄÖÜ]' for
        German)

        @type trace: C{int}
        @param trace: The level of diagnostic output to produce. C{0} doesn't
        produce any output, while C{3} will give all the output that C{megam}
        produces plus the time it took to train the model.

        @param cutoffs: Arguments specifying various conditions under
            which the training should be halted. When using C{MEGAM}, only
            C{max_iter} should be relevant. For other cutoffs see
            L{nltk.MaxentClassifier}

              - C{max_iter=v}: Terminate after C{v} iterations.
       """
        self.uppercase_letters = uppercase_letters
        self.word_freqdist = self.gen_word_freqs(train_sents)
        self.featuresets = self.gen_featsets(train_sents,
                rare_word_cutoff)
        self.features_freqdist = self.gen_feat_freqs(self.featuresets)
        self.cutoff_rare_feats(self.featuresets, rare_feat_cutoff)

        t1 = time.time()
        self.classifier = MaxentClassifier.train(self.featuresets, algorithm,
                                                 trace, **cutoffs)
        t2 = time.time()
        if trace > 0:
            print "time to train the classifier: {0}".format(round(t2-t1, 3))

    def gen_feat_freqs(self, featuresets):
        """
        Generates a frequency distribution of joint features (feature, tag)
        tuples. The frequency distribution will be used by the tagger to
        determine which (rare) features should not be considered during
        training (feature cutoff).

        This is how joint features look like::
            (('t-2 t-1', 'IN DT'), 'NN')
            (('w-2', '<START>'), 'NNP')
            (('w+1', 'of'), 'NN')

        @type featuresets: {list} of C{tuples} of (C{dict}, C{str})
        @param featuresets: a list of tuples that contain the featureset of a
        word from the training set and its POS tag.

        @rtype: C{FreqDist}
        @return: a L{frequency distribution<nltk.FreqDist()>},
        counting how often each (context information feature, tag) tuple occurs
        in the training sentences.
        """
        features_freqdist = defaultdict(int)
        for (feat_dict, tag) in featuresets:
            for (feature, value) in feat_dict.items():
                features_freqdist[ ((feature, value), tag) ] += 1
        return features_freqdist

    def gen_word_freqs(self, train_sents):
        """
        Generates word frequencies from the training sentences for the feature
        extractor.

        @type train_sents: C{list} of C{list} of tuples of (C{str}, C{str})
        @param train_sents: A list of tagged sentences.

        @rtype: C{FreqDist}
        @return: a L{frequency distribution<nltk.FreqDist()>},
        counting how often each word occurs in the training sentences.
        """
        word_freqdist = FreqDist()
        for tagged_sent in train_sents:
            for (word, _tag) in tagged_sent:
                word_freqdist[word] += 1
        return word_freqdist

    def gen_featsets(self, train_sents, rare_word_cutoff):
        """
        Generates featuresets for each token in the training sentences.

        @type train_sents: C{list} of C{list} of tuples of (C{str}, C{str})
        @param train_sents: A list of tagged sentences.

        @type rare_word_cutoff: C{int}
        @param rare_word_cutoff: Words with less occurrences than
        C{rare_word_cutoff} will be treated differently by L{extract_feats}
        than non-rare words (cf. Ratnaparkhi 1996).

        @rtype: {list} of C{tuples} of (C{dict}, C{str})
        @return:  a list of tuples that contains the featureset of
        a token and its POS-tag.
        """
        featuresets = []
        for tagged_sent in train_sents:
            history = []
            untagged_sent = untag(tagged_sent)
            for (i, (_word, tag)) in enumerate(tagged_sent):
                featuresets.append( (self.extract_feats(untagged_sent, i,
                    history, rare_word_cutoff), tag) )
                history.append(tag)
        return featuresets


    def cutoff_rare_feats(self, featuresets, rare_feat_cutoff):
        """
        Cuts off rare features to reduce training time and prevent overfitting.

        Example
        =======

            Let's say, the suffixes of this featureset are too rare to learn.

            >>> featuresets[46712]
            ({'suffix(1)': 't',
            'prefix(1)': 'L',
            'prefix(2)': 'Le',
            'prefix(3)': 'Lem',
            'suffix(3)': 'ont',
            'suffix(2)': 'nt',
            'contains-uppercase': True,
            'prefix(4)': 'Lemo',
            'suffix(4)': 'mont'},
            'NNP')

            C{cutoff_rare_feats} would then remove the rare joint features::

                (('suffix(1)', 't'), 'NNP')
                (('suffix(3)', 'ont'), 'NNP')
                ((suffix(2)': 'nt'), 'NNP')
                (('suffix(4)', 'mont'), 'NNP')

            and return a featureset that only contains non-rare features:

            >>> featuresets[46712]
            ({'prefix(1)': 'L',
            'prefix(2)': 'Le',
            'prefix(3)': 'Lem',
            'contains-uppercase': True,
            'prefix(4)': 'Lemo'},
            'NNP')


        @type featuresets: {list} of C{tuples} of (C{dict}, C{str})
        @param featuresets: a list of tuples that contain the featureset of a
        word from the training set and its POS tag

        @type rare_feat_cutoff: C{int}
        @param rare_feat_cutoff: if a (context information feature, tag)
        tuple occurs less than C{rare_feat_cutoff} times in the training
        set, then its corresponding feature will be removed from the
        C{featuresets} to be learned.
        """
        never_cutoff_features = set(['w','t'])

        for (feat_dict, tag) in featuresets:
            for (feature, value) in feat_dict.items():
                feat_value_tag = ((feature, value), tag)
                if self.features_freqdist[feat_value_tag] < rare_feat_cutoff:
                    if feature not in never_cutoff_features:
                        feat_dict.pop(feature)


    def extract_feats(self, sentence, i, history, rare_word_cutoff=5):
        """
        Generates a featureset from a word (in a sentence). The features
        were chosen as described in Ratnaparkhi (1996) and his Java
        software package U{MXPOST<ftp://ftp.cis.upenn.edu/pub/adwait/jmx>}.

        The following features are extracted:

            - features for all words: last tag (C{t-1}), last two tags (C{t-2
              t-1}), last words (C{w-1}) and (C{w-2}), next words (C{w+1}) and
              (C{w+2})
            - features for non-rare words: current word (C{w})
            - features for rare words: word suffixes (last 1-4 letters),
              word prefixes (first 1-4 letters),
              word contains number (C{bool}), word contains uppercase character
              (C{bool}), word contains hyphen (C{bool})

        Ratnaparkhi experimented with his tagger on the Wall Street Journal
        corpus (Penn Treebank project). He found that the tagger yields
        better results when words which occur less than 5 times are treated
        as rare. As your mileage may vary, please adjust
        L{rare_word_cutoff} accordingly.

        Examples
        ========

            1. This is a featureset extracted from the nonrare (word, tag)
            tuple ('considerably', 'RB')

            >>> featuresets[22356]
            ({'t-1': 'VB',
            't-2 t-1': 'TO VB',
            'w': 'considerably',
            'w+1': '.',
            'w+2': '<END>',
            'w-1': 'improve',
            'w-2': 'to'},
            'RB')

            2. A featureset extracted from the rare tuple ('Lemont', 'NN')

            >>> featuresets[46712]
            ({'suffix(1)': 't',
            'prefix(1)': 'L',
            'prefix(2)': 'Le',
            'prefix(3)': 'Lem',
            'suffix(3)': 'ont',
            'suffix(2)': 'nt',
            'contains-uppercase': True,
            'prefix(4)': 'Lemo',
            'suffix(4)': 'mont'},
            'NNP')


        @type sentence: C{list} of C{str}
        @param sentence: A list of words, usually a sentence.

        @type i: C{int}
        @param i: The index of a word in a sentence, where C{sentence[0]} would
        represent the first word of a sentence.

        @type history: C{int} of C{str}
        @param history: A list of POS-tags that have been assigned to the
        preceding words in a sentence.

        @type rare_word_cutoff: C{int}
        @param rare_word_cutoff: Words with less occurrences than
        C{rare_word_cutoff} will be treated differently than non-rare words
        (cf. Ratnaparkhi 1996).

        @rtype: C{dict}
        @return: a dictionary of features extracted from a word's
        context.
        """
        features = {}
        hyphen = re.compile("-")
        number = re.compile("\d")
        uppercase = re.compile(self.uppercase_letters)

        #get features: w-1, w-2, t-1, t-2.
        #takes care of the beginning of a sentence
        if i == 0: #first word of sentence
            features.update({"w-1": "<START>", "t-1": "<START>",
                             "w-2": "<START>", "t-2 t-1": "<START> <START>"})
        elif i == 1: #second word of sentence
            features.update({"w-1": sentence[i-1], "t-1": history[i-1],
                             "w-2": "<START>",
                             "t-2 t-1": "<START> %s" % (history[i-1])})
        else:
            features.update({"w-1": sentence[i-1], "t-1": history[i-1],
                "w-2": sentence[i-2],
                "t-2 t-1": "%s %s" % (history[i-2], history[i-1])})

        #get features: w+1, w+2. takes care of the end of a sentence.
        for inc in [1, 2]:
            try:
                features["w+%i" % (inc)] = sentence[i+inc]
            except IndexError:
                features["w+%i" % (inc)] = "<END>"

        if self.word_freqdist[sentence[i]] >= rare_word_cutoff:
            #additional features for 'non-rare' words
            features["w"] = sentence[i]

        else: #additional features for 'rare' or 'unseen' words
            features.update({"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:], "suffix(3)": sentence[i][-3:],
                "suffix(4)": sentence[i][-4:], "prefix(1)": sentence[i][:1],
                "prefix(2)": sentence[i][:2], "prefix(3)": sentence[i][:3],
                "prefix(4)": sentence[i][:4]})
            if hyphen.search(sentence[i]) != None:
                #set True, if regex is found at least once
                features["contains-hyphen"] = True
            if number.search(sentence[i]) != None:
                features["contains-number"] = True
            if uppercase.search(sentence[i]) != None:
                features["contains-uppercase"] = True

        return features


    def tag(self, sentence, rare_word_cutoff=5):
        """
        Attaches a part-of-speech tag to each word in a sequence.

        @type sentence: C{list} of C{str}
        @param sentence: a list of words to be tagged.

        @type rare_word_cutoff: C{int}
        @param rare_word_cutoff: words with less occurrences than
        C{rare_word_cutoff} will be treated differently than non-rare words
        (cf. Ratnaparkhi 1996).

        @rtype: C{list} of C{tuples} of (C{str}, C{str})
        @return: a list of tuples consisting of a word and its corresponding
        part-of-speech tag.
        """
        history = []
        for i in xrange(len(sentence)):
            featureset = self.extract_feats(sentence, i, history,
                                               rare_word_cutoff)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


def demo(corpus, num_sents):
    """
    Loads a few sentences from the Brown corpus or the Wall Street Journal
    corpus, trains them, tests the tagger's accuracy and tags an unseen
    sentence.

    @type corpus: C{str}
    @param corpus: Name of the corpus to load, either C{brown} or C{treebank}.

    @type num_sents: C{int}
    @param num_sents: Number of sentences to load from a corpus. Use a small
    number, as training might take a while.
    """
    if corpus.lower() == "brown":
        from nltk.corpus import brown
        tagged_sents = brown.tagged_sents()[:num_sents]
    elif corpus.lower() == "treebank":
        from nltk.corpus import treebank
        tagged_sents = treebank.tagged_sents()[:num_sents]
    else:
        print "Please load either the 'brown' or the 'treebank' corpus."

    size = int(len(tagged_sents) * 0.1)
    train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
    maxent_tagger = MaxentPosTagger()
    maxent_tagger.train(train_sents)
    print "tagger accuracy (test %i sentences, after training %i):" % \
        (size, (num_sents - size)), maxent_tagger.evaluate(test_sents)
    print "\n\n"
    print "classify unseen sentence: ", maxent_tagger.tag(["This", "is", "so",
        "slow", "!"])
    print "\n\n"
    print "show the 10 most informative features:"
    print maxent_tagger.classifier.show_most_informative_features(10)


if __name__ == '__main__':
    demo("treebank", 200)
    #~ featuresets = demo_debugger("treebank", 10000)
    print "\n\n\n"


