#!usr/bin/env python
# coding = utf-8

"""
    Sentiment Analysis based on supervised weight scheme (binary) for unsplited datasets by CV=10
    Used for Creating *.p Files
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
############ Load dataset ##################
print "Loading dataset ... "
path = './datasets/'
# dataset = 'rt-polarity'
# dataset = 'mpqa'
# dataset = "test"
# data_folder = [path+dataset+".pos", path+dataset+".neg"]
dataset = 'subj'
data_folder = [path+dataset+".objective", path+dataset+".subjective"]
d2v_model = doc2vec.load_docs(data_folder, clean_string=True)
print "Done!"
#############################################


########################################
print "Run algorithms, CV=10"
test_results = []
r = range(0, 10)
for i in r:
    print "Split Num = %d" % i
    d2v_model.train_test_split(i)  

    d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='1')
    # d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='12')
    # d2v_model.sws_w2v_art_fun(sws='NBSVM', ngram='123')

    # d2v_model.sws_w2v_art_fun(sws='OR', ngram='1')
    # d2v_model.sws_w2v_art_fun(sws='OR', ngram='12')
    # d2v_model.sws_w2v_art_fun(sws='OR', ngram='123')

    # d2v_model.sws_w2v_art_fun(sws='WFO', ngram='1')
    # d2v_model.sws_w2v_art_fun(sws='WFO', ngram='12')
    # d2v_model.sws_w2v_art_fun(sws='WFO', ngram='123')

test_results=[]
with open("accuracy", "rb") as f:
    for line in f:
        acc = line.strip()
        test_results.append(float(acc))
print "\nScores: ", test_results
print "Average accuracy: %f \n" % np.mean(test_results)
sys.stdout.flush()
os.remove("accuracy")

pickle_d2v_model = path + dataset + "-d2vmodel-2.p" 
with open(pickle_d2v_model, "wb") as f:
    pickle.dump([d2v_model], f)     # create a pickle object
print "d2v_model *.P File created!"





