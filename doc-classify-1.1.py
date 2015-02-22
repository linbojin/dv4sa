#!/usr/bin/env python
# coding=utf-8

"""doc2vec version 1.1.0 """

import doc2vec
import numpy as np
from multiprocessing import Pool
import os, sys
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
try:
    import cPickle as pickle
except ImportError:
    import pickle



print "Doc2vec version 1.2.0"
print "Loading documents... "
##############################
path = './datasets/'
dataset = 'rt-polarity'
#dataset = 'mpqa'
# dataset = "test"
data_folder = [path+dataset+".pos", path+dataset+".neg"]
d2v_model = doc2vec.load_docs(data_folder, clean_string=True)
print "Done!"
###############################
print "Loading wordvecs... "
w2v_file = './datasets/wordvecs/GoogleNews-vectors-negative300.bin'
# w2v_file = './datasets/wordvecs/vectors.bin'
w2v_model = doc2vec.load_word_vec(w2v_file, d2v_model.vocab)
print "Done!"
##############################
print "loading word_centroid_map ..."
centroid_map = path + dataset + "-centroid-map.p"
x = pickle.load(open(centroid_map, "rb"))
w2v_model.word_centroid_map = x[0]
print "Done!"

###############################
print "cross validation"
train_results=[]
test_results = []
c=0.1

r = range(0, 10)
for i in r:
    print "cv = %d" % i
    d2v_model.train_test_split(i)  #  len(train) = 9587 len(test) = 1084
    d2v_model.count_data()
    # d2v_model.get_bag_of_words_sklearn()           # 77.1 c=1  tf-idf weight scheme in sklearn
    # d2v_model.get_bag_of_words(cre_adjust=False)   # 77.2 c=1  custom tf-idf 
    # d2v_model.get_bag_of_words(cre_adjust=True)    # 77.5 c=1  cre tf-idf weight
    
    # d2v_model.get_avg_feature_vecs(w2v_model)      # 77.3 c=1  word vec average scheme
    # d2v_model.get_tf_idf_feature_vecs(w2v_model)   # 77.2 c=1  word vec tf-idf scheme
    # d2v_model.get_tf_idf_feature_vecs(w2v_model, cre_adjust=True)  # 76.9 c=1  cre tf-idf word vec average scheme

    d2v_model.create_bag_of_centroids(w2v_model.word_centroid_map) # 75.3 c=0.1 word vec bag of centroids
    d2v_model.create_bag_of_centroids(w2v_model.word_centroid_map, cre_adjust=True) 

    text_clf = LinearSVC(C=c)
    _ = text_clf.fit(d2v_model.train_doc_vecs, d2v_model.train_labels)
    perf = text_clf.score(d2v_model.test_doc_vecs, d2v_model.test_labels) 
    perf2 = text_clf.score(d2v_model.train_doc_vecs, d2v_model.train_labels)
    print " Train accuracy:" + str(perf2)
    print " Test accuracy:" + str(perf)

    # clf = SVC(kernel='rbf',C=c, gamma=g).fit(train, train_Label)
    # perf2 = clf.score(train, train_Label)
    # perf = clf.score(test, test_Label)
    # print " cv: " + str(i) + ", perf->train: " + str(perf2)
    # print " cv: " + str(i) + ", perf->test: " + str(perf)
    print 
    train_results.append(perf2)
    test_results.append(perf)

print "****** (c=%f) ******" % c
print "   Train Average accuracy: %f" % np.mean(train_results)
print "   Test  Average accuracy: %f \n" % np.mean(test_results)
sys.stdout.flush()
###############################################



# #
# d2v_model.cre_sim_doc_vecs(w2v_model)
# training(2)

# #
# print "Calculating tf-idf Vectors"
# d2v_model.get_tf_idf_feature_vecs(w2v_model, cre_adjust=False)   # 0.71
# print "Done!"
# training(14)
# # #
# print "Calculating cre adjust tf-idf Vectors"
# d2v_model.get_tf_idf_feature_vecs(w2v_model, cre_adjust=True)   # 0.71
# print "Done!"
# training(16)
# # #
# print "Calculate Average Vectors"  # 0.776   0.787
# d2v_model.get_avg_feature_vecs(w2v_model)
# print "Done!"
# training(4)
# #
# #
# print "\nCalculating Bag of words vectors"
# c=1.2
# cross_validation(d2v_model, c)
# # d2v_model.get_bag_of_words(cre_adjust=False)   # c=0.1 0.769   False 0.761
# # cre_weight = [ 1.00000072  1.01045179  1.00366998 ...,  1.36067581  1.36067581 1.36067581]
# # min_count = 2 [ 1.00000072  1.01045179  1.00366998 ...,  1.09403062  1.09403062 1.09403062]
# print "Done!"
# # training(0.1)
#
# print "Calculating cre_adjust Bag of words vectors"
# d2v_model.get_bag_of_words(cre_adjust=True)   # c=0.1 0.769   False 0.761
# # cre_weight = [ 1.00000072  1.01045179  1.00366998 ...,  1.36067581  1.36067581 1.36067581]
# # min_count = 2 [ 1.00000072  1.01045179  1.00366998 ...,  1.09403062  1.09403062 1.09403062]
# print "Done!"
# training(0.1)
#
# #
# print "Calculating bag_of_centroids"
# #w2v_model.get_w2v_centroid()
# d2v_model.create_bag_of_centroids(w2v_model.word_centroid_map,cre_adjust=False)
# print "Done!"
# training(0.1)
#
# #
# print "Calculating cre_adjust bag_of_centroids"
# d2v_model.create_bag_of_centroids(w2v_model.word_centroid_map,cre_adjust=True)
# print "Done!"
# training(0.1)


# with open("word_centroid_map.p", "wb") as f:
#     pickle.dump([w2v_model.word_centroid_map], f)     # create a pickle object
# print "dataset created!"


#
# print "Starting Cross Validation on Linear SVM ..."
# c=2
# g=2
# cross_validation(c,g)  # (2,2) 0.799
#



