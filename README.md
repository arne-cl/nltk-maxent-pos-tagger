nltk-maxent-pos-tagger
======================

`nltk-maxent-pos-tagger` is a part-of-speech (POS) tagger based on Maximum
Entropy (ME) principles written for [NLTK](http://nltk.org/ "Python's Natural Language Toolkit").
It is based on NLTK's Maximum Entropy classifier
(`nltk.classify.maxent.MaxentClassifier`), which uses
[MEGAM](http://hal3.name/megam "Hal Daume's MEGA Model Optimization Package") for number
crunching.

Part-of-Speech Tagging
----------------------

`nltk-maxent-pos-tagger` uses the set of features proposed by 
[Ratnaparki (1996)](http://www.aclweb.org/anthology-new/W/W96/W96-0213.pdf 
"A Maximum Entropy Model for Part-of-Speech Tagging"), which are also used 
in his [MXPOST](ftp://ftp.cis.upenn.edu/pub/adwait/jmx/) implementation (Java).


Installation
------------

1.  Install Python and NLTK.

NLTK offers lots of data sets, which you might download and install from within
a Python shell:

    import nltk
    nltk.download()

Download at least `brown` or `treebank`, as nltk-maxent-pos-tagger uses them
for its `demo()` function.

2. (Mac) Install MEGAM.

On Mac, it is easy to install MEGAM using brew:

    brew tap homebrew/science
    brew install megam

Usage
-----

Have a look at the example given in the `demo()` function in `mxpost.py`.
Basically, you just have to import the tagger and train it with labelled data
to use it:

    import mxpost
    maxent_tagger = mxpost.MaxentPosTagger()
    maxent_tagger.train(tagged_training_sentences)

    for sentence in unlabeled_sentences:
        maxent_tagger.tag(sentence)


Meta
----

Status: Beta. I wrote this in 2008 as a semester project for a class on NLP tools.  
Licence: GPL Version 3  
Original Author: Arne Neumann  
Contributors: Arne Neumann, Andrew Drozdov


TODO
----

1.   *speed / memory consumption*   
     As you can expect, a Python implementation is much slower and consumes
     much more RAM than similar tools written in Java or C/C++ (MXPOST,
     acopost, C&C etc.). This being said, most of the time isn't spend in
     Python but rather in MEGAM (which is written in O'Caml and therefore
     shouldn't have such issues).  NLTK currently is only able to encode POS
     features explicitly when converting data for MEGAM. According to the MEGAM
     website, using implicit feature encoding should be much faster.
    
2.  *accuracy*  
    I trained several taggers on the WSJ corpus (90% training / 10% test data).
    nltk-maxent-pos-tagger achieved an accuracy of 93.64% (100 iterations, rare
    feature cutoff = 5) while MXPOST reached 96.93% (100 iterations). Since
    both implementations use the same feature set, results shouldn't be that
    different.  Unfortunately, there's no source code available for `MXPOST`,
    but comparing `nltk-maxent-pos-tagger` with OpenNLP's implementation should
    be helpful.  

