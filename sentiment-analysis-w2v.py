#!usr/bin/env python
# coding = utf-8

"""
    Sentiment Analysis based on word2vec
    Author: Linbo
    Date: 15.03.2015
"""

import os
import sys
import re
import numpy as np
import doc2vec
from sklearn.svm import LinearSVC
try:
    import cPickle as pickle
except ImportError:
    import pickle

print "Sentiment Analysis based on Word2vec"
############ Load d2v_model and dataset from *.p Files##################
print "Loading d2v_model and dataset from *.p File"
path = './datasets/'
dataset = 'rt-polarity'
# dataset = 'mpqa'
# dataset = "test"
# dataset = 'subj'
pickle_d2v_model = path + dataset + "-d2vmodel.p" 
x = pickle.load(open(pickle_d2v_model, "rb"))
d2v_model = x[0]
print "Done!"
#############################################

############ Load wordvecs #################
print "Loading wordvecs... "
w2v_file = './datasets/wordvecs/GoogleNews-vectors-negative300.bin'
# w2v_file = './datasets/wordvecs/vectors.bin'
w2v_model = doc2vec.load_word_vec(w2v_file, d2v_model.vocab)
print "Done!"

############################################

def Convert_data_format(vecs, labels, outfn):
    output = []
    for vec, label in zip(vecs, labels):
        line = [str(label)]
        for i, value in enumerate(vec):
            line += ["%i:%f" % (i+1 , value)]
        output += [" ".join(line)]
    output = "\n".join(output)

    with open(outfn, "w") as f:
        f.writelines(output)

#############################################
print "Run Algorithms CV=10"
train_results=[]
test_results = []

r = range(0, 10)
for i in r:
    print "Split Num = %d" % i
    d2v_model.train_test_split(i)  #  len(train) = 9587 len(test) = 1084
    d2v_model.count_data()

    # d2v_model.get_avg_feature_vecs(w2v_model)        # 77.6 c=1  word vec average scheme
    # d2v_model.get_tf_idf_feature_vecs(w2v_model)      # 77.3 c=1  word vec tf-idf scheme
    # d2v_model.get_tf_idf_feature_vecs(w2v_model, cre_adjust=True)  # 76.9 c=1  word vec cre tf-idf scheme

    d2v_model.get_sws_w2v_feature_vecs(w2v_model)


    Convert_data_format(d2v_model.train_doc_vecs, d2v_model.train_labels, "train-swsvecs.txt")
    Convert_data_format(d2v_model.test_doc_vecs, d2v_model.test_labels, "test-swsvecs.txt")

    liblinear='liblinear-1.96'
    trainsvm = os.path.join(liblinear, "train")
    predictsvm = os.path.join(liblinear, "predict")
    # os.system(trainsvm + " -s 0 train-swsvecs.txt model.logreg")
    # os.system(predictsvm + " -b 1 test-swsvecs.txt model.logreg " + './outputs/W2V-TEST')
    os.system(trainsvm + " train-swsvecs.txt model.logreg")
    os.system(predictsvm + " test-swsvecs.txt model.logreg " + './outputs/W2V-TEST')
    os.remove("model.logreg")
    os.remove("train-swsvecs.txt")
    os.remove("test-swsvecs.txt")

    print


with open("accuracy", "rb") as f:
    for line in f:
        acc = line.strip()
        test_results.append(float(acc))
print "\nScores: ", test_results
print "Test  Average accuracy: %f \n" % np.mean(test_results)
sys.stdout.flush()
os.remove("accuracy")


#     text_clf = LinearSVC(C=1)
#     _ = text_clf.fit(d2v_model.train_doc_vecs, d2v_model.train_labels)
#     perf = text_clf.score(d2v_model.test_doc_vecs, d2v_model.test_labels) 
#     perf2 = text_clf.score(d2v_model.train_doc_vecs, d2v_model.train_labels)
#     print " Train accuracy:" + str(perf2)
#     print " Test accuracy:" + str(perf)

#     # clf = SVC(kernel='rbf',C=c, gamma=g).fit(train, train_Label)
#     # perf2 = clf.score(train, train_Label)
#     # perf = clf.score(test, test_Label)
#     # print " cv: " + str(i) + ", perf->train: " + str(perf2)
#     # print " cv: " + str(i) + ", perf->test: " + str(perf)
#     print 
#     train_results.append(perf2)
#     test_results.append(perf)

# print "****** Average accuracy ******" 
# print "   Train Average accuracy: %f" % np.mean(train_results)
# print "   Test  Average accuracy: %f \n" % np.mean(test_results)
# sys.stdout.flush()



























