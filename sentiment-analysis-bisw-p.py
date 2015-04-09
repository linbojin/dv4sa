#!usr/bin/env python
# coding = utf-8

"""
    Sentiment Analysis based on supervised weight scheme (binary) for unsplited datasets by CV=10
        NBSVM
        OR
        WFO
    Author: Linbo
    Date: 15.03.2015
"""

import os
import sys
import numpy as np
import doc2vec
try:
    import cPickle as pickle
except ImportError:
    import pickle

print "Sentiment Analysis based on Supervied Weight Scheme"
############ Load d2v_model and dataset from *.p Files##################
print "Loading d2v_model and dataset from *.p File"
path = './datasets/'
# dataset = 'rt-polarity'
dataset = 'mpqa'
# dataset = "test"
# dataset = 'subj'
# dataset = 'pl2k'
pickle_d2v_model = path + dataset + "-d2vmodel.p" 
x = pickle.load(open(pickle_d2v_model, "rb"))
d2v_model = x[0]
print "Done!"
#############################################

# print "Loading wordvecs... "
# w2v_file = './datasets/wordvecs/GoogleNews-vectors-negative300.bin'
# # w2v_file = './datasets/wordvecs/vectors.bin'
# w2v_model = doc2vec.load_word_vec(w2v_file, d2v_model.vocab)
# print "Done!"
w2v_model = None
########################################
print "Run algorithms, CV=10"
test_results = []
r = range(0, 10)
for i in r:
    print "Split Num = %d" % i
    d2v_model.train_test_split(i)  
    
    d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='1', w2v=w2v_model)
    # d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='12', w2v=w2v_model) #[91.0995, 87.963, 88.5, 89.3204, 87.5648, 92.7461, 90.3382, 92.891, 93.6842, 83.9378]
    # d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='123')

    # d2v_model.sws_w2v_art_fun(sws='OR', ngram='1', w2v=w2v_model)
    # d2v_model.sws_w2v_art_fun(sws='OR', ngram='12', w2v=w2v_model)   79.5%
    # d2v_model.sws_w2v_art_fun(sws='OR', ngram='123')

    # d2v_model.sws_w2v_art_fun(sws='WFO', ngram='1')
    # d2v_model.sws_w2v_art_fun(sws='WFO', ngram='12')
    # d2v_model.sws_w2v_art_fun(sws='WFO', ngram='123')

    # d2v_model.sws_w2v_vectors()   # lowest speed !!!

test_results=[]
with open("accuracy", "rb") as f:
    for line in f:
        acc = line.strip()
        test_results.append(float(acc))
print "\nScores: ", test_results
print "Average accuracy: %f \n" % np.mean(test_results)
sys.stdout.flush()
os.remove("accuracy")




