#!usr/bin/env python
# coding = utf-8

"""
    Sentiment Analysis based on bag_of_words for unsplited datasets by CV=10
        sklearn tf-idf
        customed bag_of_words
        cred-tfidf
    Author: Linbo
    Date: 15.03.2015
"""

import doc2vec
import numpy as np
from sklearn.svm import LinearSVC

print "Sentiment Analysis based on Bag of words"
############ Load dataset ##################
print "Loading dataset ... "
path = './datasets/'
dataset = 'rt-polarity'
dataset = 'mpqa'
data_folder = [path+dataset+".pos", path+dataset+".neg"]
# dataset = 'subj'
# data_folder = [path+dataset+".objective", path+dataset+".subjective"]
d2v_model = doc2vec.load_docs(data_folder, clean_string=True)
print "Done!"
#############################################


#######################################
print "Run algorithms, CV=10"
train_results=[]
test_results = []
c=1
r = range(0, 10)
for i in r:
    print "cv = %d" % i
    d2v_model.train_test_split(i)  
    d2v_model.count_data()
    # d2v_model.get_bag_of_words_sklearn()           # 77.1 c=1  tf-idf weight scheme in sklearn
    # d2v_model.get_bag_of_words(cre_adjust=False)   # 77.2 c=1  custom tf-idf 
    d2v_model.get_bag_of_words(cre_adjust=True)    # 77.5 c=1  custom cre tf-idf weight
    
    text_clf = LinearSVC(C=c)
    _ = text_clf.fit(d2v_model.train_doc_vecs, d2v_model.train_labels)
    perf = text_clf.score(d2v_model.test_doc_vecs, d2v_model.test_labels) 
    perf2 = text_clf.score(d2v_model.train_doc_vecs, d2v_model.train_labels)
    print " Train accuracy:" + str(perf2)
    print " Test accuracy:" + str(perf)

    print 
    train_results.append(perf2)
    test_results.append(perf)

print "****** (c=%f) ******" % c
print "   Train Average accuracy: %f" % np.mean(train_results)
print "   Test  Average accuracy: %f \n" % np.mean(test_results)

