#!usr/bin/env python
# coding = utf-8

"""
    Sentiment Analysis based on supervised weight scheme (binary) for splited datasets
        NBSVM
        OR
        WFO
    Author: Linbo
    Date: 15.03.2015
"""

import os
import numpy as np
import doc2vec

print "Sentiment Analysis based on Supervied Weight Scheme"
############ Load dataset ##################
print "Loading dataset ... "
path = './datasets/'
data_folder = [path+"aclImdb-train-pos.txt",path+"aclImdb-train-neg.txt",
               path+"aclImdb-test-pos.txt",path+"aclImdb-test-neg.txt"]

d2v_model = doc2vec.load_docs(data_folder, clean_string=False, splited=True)
print "Done!"
#############################################

# print "Loading wordvecs... "
# w2v_file = './datasets/wordvecs/GoogleNews-vectors-negative300.bin'
# # w2v_file = './datasets/wordvecs/vectors.bin'
# w2v_model = doc2vec.load_word_vec(w2v_file, d2v_model.vocab)
# print "Done!"
w2v_model = None
########################################
print "Run algorithms"

d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='1', w2v=w2v_model)
# d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='12', w2v=w2v_model)
# d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='123')

# d2v_model.sws_w2v_art_fun(sws='OR', ngram='1', w2v=w2v_model)
# d2v_model.sws_w2v_art_fun(sws='OR', ngram='12', w2v=w2v_model)
# d2v_model.sws_w2v_art_fun(sws='OR', ngram='123')

# d2v_model.sws_w2v_art_fun(sws='WFO', ngram='1', w2v=w2v_model)
# d2v_model.sws_w2v_art_fun(sws='WFO', ngram='12', w2v=w2v_model)
# d2v_model.sws_w2v_art_fun(sws='WFO', ngram='123')

os.remove("accuracy")




