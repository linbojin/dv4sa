#!/usr/bin/env python
# coding=utf-8

"""doc2vec version 1.1.0 """

import doc2vec
import numpy as np
from multiprocessing import Pool
import os, sys
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


def make_idx_data_cv(d2v, cv):
    train, test, train_Label, test_Label = [], [], [], []
    d2v.train_data = []
    d2v.test_data = []
    for rev in d2v.revs:
        doc_vec = rev["doc_vec"]
        if rev["split"] == cv:        # 1/10 for testing and 9/10 for training
            d2v.test_data.append(rev)
            test.append(doc_vec)
            test_Label.append(rev['y'])
        else:
            d2v.train_data.append(rev)
            train.append(doc_vec)
            train_Label.append(rev['y'])
    train = np.array(train, dtype="float32")
    test = np.array(test, dtype="float32")
    return train, test, train_Label, test_Label


def cross_validation(c,g):
        train_results=[]
        test_results = []
        r = range(0, 10)

        for i in r:
            train, test, train_Label, test_Label = make_idx_data_cv(d2v_model, i)  #  len(train) = 9587 len(test) = 1084
            train, test = d2v_model.get_bag_of_words()

            text_clf = LinearSVC(C=c)
            _ = text_clf.fit(train, train_Label)
            perf = text_clf.score(test, test_Label)
            perf2 = text_clf.score(train, train_Label)
            print "cv: " + str(i) + ", perf->train: " + str(perf2)
            print "cv: " + str(i) + ", perf->test: " + str(perf)

            # clf = SVC(kernel='rbf',C=c, gamma=g).fit(train, train_Label)
            # perf2 = clf.score(train, train_Label)
            # perf = clf.score(test, test_Label)
            # print " cv: " + str(i) + ", perf->train: " + str(perf2)
            # print " cv: " + str(i) + ", perf->test: " + str(perf)

            train_results.append(perf2)
            test_results.append(perf)

        print "****** (c=%f,g=%f) ******" % (c,g)
        print "   Train Average accuracy: %f" % np.mean(train_results)
        print "   Test  Average accuracy: %f \n" % np.mean(test_results)
        sys.stdout.flush()


def training(c=1):
    train_results = []
    test_results = []
    text_clf = LinearSVC(C=c)
    _ = text_clf.fit(d2v_model.train_doc_vecs, d2v_model.train_labels)
    perf2 = text_clf.score(d2v_model.train_doc_vecs, d2v_model.train_labels)
    perf = text_clf.score(d2v_model.test_doc_vecs, d2v_model.test_labels)

    train_results.append(perf2)
    test_results.append(perf)

    print "****** c=%f ******" % (c)
    print "   Train Average accuracy: %f" % np.mean(train_results)
    print "   Test  Average accuracy: %f \n" % np.mean(test_results)

print "Doc2vec version 1.0.0"
print "Loading documents... "
path = './datasets/'
data_folder = [path+"rt-polarity.pos",path+"rt-polarity.neg"]
d2v_model = doc2vec.load_docs(data_folder, clean_string=True)
d2v_model.train_test_split(test_size=0.1)
d2v_model.count_data()
print "Done!"
#
#
print "Loading wordvecs... "
w2v_file = './datasets/wordvecs/GoogleNews-vectors-negative300.bin'
#w2v_file = './datasets/wordvecs/vectors.bin'
w2v_model = doc2vec.load_word_vec(w2v_file, d2v_model.vocab)
print "Done!"
# #
# #
print "Calculate Average Vectors"  # 0.776   0.787
d2v_model.get_avg_feature_vecs(w2v_model)
print "Done!"
training(2)
#
#
print "Calculating tf-idf Vectors"
d2v_model.get_tf_idf_feature_vecs(w2v_model, cre_adjust=False)   # 0.71
print "Done!"
training(2)

print "Calculating Bag of words vectors"
d2v_model.get_bag_of_words(cre_adjust=True)   # c=0.1 0.769   False 0.761
# cre_weight = [ 1.00000072  1.01045179  1.00366998 ...,  1.36067581  1.36067581 1.36067581]
# min_count = 2 [ 1.00000072  1.01045179  1.00366998 ...,  1.09403062  1.09403062 1.09403062]
print "Done!"
training(0.1)

#
print "Calculating bag of create_bag_of_centroids"
w2v_model.get_w2v_centroid()
d2v_model.create_bag_of_centroids(w2v_model.word_centroid_map)
print "Done!"
training()
#
# print "Starting Cross Validation on Linear SVM ..."
# c=2
# g=2
# cross_validation(c,g)  # (2,2) 0.799
#


