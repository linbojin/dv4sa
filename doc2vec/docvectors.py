#!/usr/bin/env python
# coding=utf-8

"""
    docvector.py
    Author: Linbo
    Date: 15.03.2015
"""

import os
import numpy as np
from collections import defaultdict
from doc2vec.utility import *
# from nltk.corpus import stopwords
import logging
logging.basicConfig(level=logging.INFO)
# from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer   # count the occurrences words and build a matrix (doc X vocab)
from sklearn.feature_extraction.text import TfidfTransformer  # transfer occurrences into tf or tfidf
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import nbsvm 
# from stemming.porter2 import stem
# from nltk.stem.porter import *

class DocVectors(object):

    def __init__(self, revs, labels, vocab, cv_split, clusters=None):
        """
        Initialize a DocVectors class based on vocabulary and vectors

        Parameters
        ----------
        vocab : np.array
            1d array with the vocabulary
        vectors : np.array
            2d array with the vectors calculated by word2vec
        clusters : word2vec.WordClusters (optional)
            1d array with the clusters calculated by word2vec
        """
        self.revs = revs
        self.labels = labels
        self.cv_split = cv_split

        self.train_data = []
        self.train_labels =[]
        self.test_data = []
        self.test_labels =[]

        self.train_word_nums = 0.0
        self.pos_words_num = 0.0
        self.neg_words_num = 0.0

        self.vocab = vocab
        self.train_vocab = {}
        self.train_tf = []
        self.test_tf = []
        self.pos_counts = {}
        self.neg_counts = {}
        self.clusters = clusters

        self.weight_dict = {}
        self.sentiment_vocab={}

############## New Function ################
    def dist_weight(self):
        # print " Creating cre_adjust_tf_idf weight"
        vocab_list = self.get_train_vocab_list()
        # dist_weight = np.zeros(len(self.train_vocab), dtype="float32")
        # weight_dict = dict()
        N_pos = float(len([label for label in self.train_labels if label == 1]) + 1 )
        N_neg = float(len([label for label in self.train_labels if label == 0]) + 1 )
        
        for w in vocab_list:
            Fp = (self.pos_counts[w] + 1) / N_pos
            Fn = (self.neg_counts[w] + 1) / N_neg
            cre_weight = ( Fn **2 +Fp**2) / (Fn+ Fp)**2
            # cre_weight =((self.neg_counts[w])**2 +(self.pos_counts[w])**2) / (self.train_vocab[w])**2
            or_weight = abs(Fp*(1-Fn) / (Fn*(1-Fp)))
            lam = 0.1
            wfo_weight = abs( Fp**lam * math.log( (Fp/Fn)**(1-lam)) )
            # i = vocab_list.index(w)
            # dist_weight[i] = cre_weight * or_weight
            self.weight_dict[w] = cre_weight

        reverse_weight_list = self.weight_dict.keys()
        reverse_weight_list.sort(self.weight_sort_fun)

        print "reverse_weight_list"
        for word in reverse_weight_list:
           print word, ':', self.weight_dict[word] 

    def sentiment_sort_fun(self,x,y):
        return cmp(self.sentiment_vocab[y], self.sentiment_vocab[x])

    def get_sentiment_vocab_list(self):
        """
            get the train word list based on reverse count
        """
        reverse_vocab_list = self.sentiment_vocab.keys()
        reverse_vocab_list.sort(self.sentiment_sort_fun)

        #print reverse_vocab_list
        # for word in reverse_vocab_list:
        #    print word, ':', self.vocab[word],' ',
        return reverse_vocab_list

    def new_fun(self):
        sentiment_vocab = {}
        weight_dict = self.weight_dict
        self.dist_weight()
        for word in weight_dict:
            if weight_dict[word] > 0.55:
                sentiment_vocab[word]=weight_dict[word]
        self.sentiment_vocab = sentiment_vocab

    def build_new_tf_matrix(self, w2v, revs, tf_data):
        doc_num = len(revs)
        # num_vocab = len(self.vocab)
        # num_vocab = len(self.train_vocab)
        num_vocab = len(self.sentiment_vocab)
        tf_matrix = np.zeros((doc_num, num_vocab), dtype="float32")

        counter = 0
        # vocab_list = self.get_vocab_list()
        # vocab_list = self.get_train_vocab_list()
        vocab_list = self.get_sentiment_vocab_list()
        test = 0
        for rev in revs:
            tf = tf_data[counter]
            tf_list = np.zeros(num_vocab, dtype="float32")
            for w in tf:
                if w in vocab_list:
                    try:
                        i = vocab_list.index(w)
                        tf_list[i] = tf[w]
                    except ValueError:
                        # Ignore out-of-vocabulary items
                        sim_words = w2v.most_similar(w)
                        for sim_word in sim_words:
                            if sim_word[0] in vocab_list and sim_word[1] > 0.7:  # 
                                # if sim_word[1] < 0.5:
                                #     print w, sim_word
                                i = vocab_list.index(sim_word[0])
                                tf_list[i] = tf[w]
                                break
                            else:
                                continue
                else:
                    sim_words = w2v.most_similar(w)
                    for sim_word in sim_words:
                        if sim_word[0] in vocab_list and sim_word[1] > 0.6:  # 
                            # if sim_word[1] < 0.5:
                            #     print w, sim_word
                            i = vocab_list.index(sim_word[0])
                            tf_list[i] = tf[w]
                            break
                        else:
                            continue

            if np.array_equal(tf_list, np.zeros(num_vocab, dtype="float32")):
                print "Zero tf array for test doc"
                print rev
                #tf_list = np.random.uniform(0,2, num_vocab)
                #print tf_list
            
            # # normalize tf list
            # tf_norm_list = 0.5 + 0.5 * tf_list / np.amax(tf_list)
            # print tf_norm_list
            # tf_matrix[counter] = tf_norm_list

            tf_matrix[counter] = tf_list
            counter = counter + 1

        return tf_matrix

    def get_new_bag_of_words(self, w2v, cre_adjust=False):
        """
            tf-idf weight scheme
        """
       # print " Computing train idf tf matrix..."
        train_tf_matrix = self.build_new_tf_matrix(w2v, self.train_data, self.train_tf)
        # print train_tf_matrix[0]
        # print self.get_vocab_list()
        if cre_adjust:
            cre_adjust_weight = self.cre_adjust_tf_idf()
            train_tf_matrix = train_tf_matrix * cre_adjust_weight
        transformer = TfidfTransformer(sublinear_tf=cre_adjust) # sublinear_tf=True, selected for cre_adjust = true
        train_tfidf_matrix = transformer.fit_transform(train_tf_matrix)
        self.train_doc_vecs = train_tfidf_matrix

        #print " Computing test idf tf matrix..."
        test_tf_matrix = self.build_new_tf_matrix(w2v, self.test_data, self.test_tf)
        if cre_adjust:
            test_tf_matrix = test_tf_matrix * cre_adjust_weight
        test_tfidf_matrix = transformer.transform(test_tf_matrix)
        self.test_doc_vecs = test_tfidf_matrix
############################################









####### XXXXXXXXXXXXXXXXXXXXXXXXX ############
    def build_sim_tf_matrix(self, revs, tf_data):
        doc_num = len(revs)
        num_vocab = len(self.vocab)
        # num_vocab = len(self.train_vocab)
        tf_matrix = np.zeros((doc_num, num_vocab), dtype="float32")

        counter = 0
        vocab_list = self.get_vocab_list()
        # vocab_list = self.get_train_vocab_list()

        for rev in revs:
            tf = tf_data[counter]
            tf_list = np.zeros(num_vocab, dtype="float32")
            for w in tf:
                try:
                    i = vocab_list.index(w)
                    tf_list[i] = tf[w]
                except ValueError:
                    # Ignore out-of-vocabulary items
                    continue
            if np.array_equal(tf_list, np.zeros(num_vocab, dtype="float32")):
                print "Zero tf array for test doc"
                print rev
                #tf_list = np.random.uniform(0,2, num_vocab)
                #print tf_list
            
            # # normalize tf list
            # tf_norm_list = 0.5 + 0.5 * tf_list / np.amax(tf_list)
            # print tf_norm_list
            # tf_matrix[counter] = tf_norm_list

            tf_matrix[counter] = tf_list
            counter = counter + 1

        return tf_matrix

    def cre_sim_weight(self, w2v):
        num_centroids = max(w2v.word_centroid_map.values()) + 1
        cluster_vocab = np.zeros(num_centroids, dtype="float32")
        cluster_neg_counts = defaultdict(float)
        cluster_pos_counts = defaultdict(float)
        for word in self.train_vocab:
            index = w2v.word_centroid_map[word]
            cluster_vocab[index] += self.train_vocab[word]
            if word in self.pos_counts:
                cluster_pos_counts[index] += self.pos_counts[word]
            if word in self.neg_counts:
                cluster_neg_counts[index] += self.neg_counts[word]

        gamma = 0.8
        s_hat_average = 0.0
        cre_sim_clu_weight= np.zeros(num_centroids, dtype="float32")
        for cluster in xrange(num_centroids):
            if cluster_vocab[cluster]>0:  # cluster_vocab=[1,4,5,0,6]
                s_hat = ((cluster_neg_counts[cluster])**2 +(cluster_pos_counts[cluster])**2) / (cluster_vocab[cluster])**2
                s_hat_average += (cluster_vocab[cluster] * s_hat) / self.train_word_nums

        for cluster in xrange(num_centroids):
            if cluster_vocab[cluster]>0:
                s_bar = ((cluster_neg_counts[cluster])**2 +(cluster_pos_counts[cluster])**2 + s_hat_average * gamma) / ((cluster_vocab[cluster])**2 + gamma)
                cre_sim_clu_weight[cluster] = 0.5 + s_bar

        logging.debug("For function cre_sim_weight() **********" )
        logging.debug("s_hat_average %f" % s_hat_average)

        print "Creating cre_sim_word_weight"
        cre_sim_word_weight = np.zeros(len(self.vocab), dtype="float32")
        # cre_sim_word_weight = np.zeros(len(self.train_vocab), dtype="float32")
        zero_clusters = np.where(cre_sim_clu_weight == 0.0)[0]
        print "%d zero_clusters" % len(zero_clusters)

        vocab_list = self.get_vocab_list()
        # vocab_list = self.get_train_vocab_list()
        for w in vocab_list:
            index = vocab_list.index(w)
            cluster = w2v.word_centroid_map[w]
            # if test word in unknown cluster
            if cluster in zero_clusters:
                # print "unknown cluster test word:", w
                sim_words = w2v.most_similar(w)
                # print 'similar words:', sim_words
                for sim_word in sim_words:
                    clu = w2v.word_centroid_map[sim_word]
                    if clu in zero_clusters:
                        continue
                    else:
                        cluster = clu
                        # print "substitution word:", sim_word
                        # print "**********************************"
                        break

            Cluster = "Cluster " + cluster.astype(str)
            cre_sim_word_weight[index] = cre_sim_clu_weight[cluster] / len(w2v.clusters_dict[Cluster])

        return cre_sim_word_weight

    def cre_sim_doc_vecs(self, w2v):
        train_tf_matrix = self.build_sim_tf_matrix(self.train_data, self.train_tf)

        cre_sim_word_weight = self.cre_sim_weight(w2v)
        train_tf_matrix = train_tf_matrix * cre_sim_word_weight

        transformer = TfidfTransformer(sublinear_tf=True)
        train_tfidf_matrix = transformer.fit_transform(train_tf_matrix)

        #print " Computing train tf_idf_doc_vecs..."
        self.train_doc_vecs = self.compute_tf_idf_feature_vecs(w2v, train_tfidf_matrix)

        #print " Computing test tf matrix..."
        test_tf_matrix = self.build_sim_tf_matrix(self.test_data, self.test_tf)
        test_tf_matrix = test_tf_matrix * cre_sim_word_weight
        #print " Computing test tf_idf_doc_vecs..."
        test_tfidf_matrix = transformer.transform(test_tf_matrix)
        self.test_doc_vecs = self.compute_tf_idf_feature_vecs(w2v, test_tfidf_matrix)

















































### word vecs supervised weight scheme ######
    def compute_w2v_weight_vector(self, alpha=1):
        vocab_list = self.get_train_vocab_list()
        d = len(vocab_list)
        Fp, Fn = np.ones(d) * alpha , np.ones(d) * alpha
        for w in vocab_list:
            Fp[vocab_list.index(w)] += self.pos_counts[w]
            Fn[vocab_list.index(w)] += self.neg_counts[w]
        Fw = Fp + Fn
        Fp /= abs(Fp).sum()
        Fn /= abs(Fn).sum()

        meaning_words = 200
        # #### nbsvm ###
        # nbsvm_ratio = np.log(Fp/Fn)
        # self.weight_vector = np.abs(nbsvm_ratio)
               
        # counter = 0
        # for wt in self.weight_vector:
        #     self.weight_dict[vocab_list[counter]] = wt
        #     counter += 1

        # reverse_weight_list = self.weight_dict.keys()
        # reverse_weight_list.sort(self.weight_sort_fun)

        # nbsvm_100 =  reverse_weight_list[:meaning_words] + reverse_weight_list[-1*meaning_words:]

        # ######## WFO #######
        # lam = 0.01  # 0.1
        # wfo = Fp**lam * np.log( (Fp/Fn)**(1-lam))
        # self.weight_vector = np.abs(wfo)

        # counter = 0
        # self.weight_dict={}
        # for wt in self.weight_vector:
        #     self.weight_dict[vocab_list[counter]] = wt
        #     counter += 1

        # reverse_weight_list = self.weight_dict.keys()
        # reverse_weight_list.sort(self.weight_sort_fun)

        # wfo_100 =  reverse_weight_list[:meaning_words] + reverse_weight_list[-1*meaning_words:]

        # #### odds rotio #####

        OR = np.log(Fp*(1-Fn)/(Fn*(1-Fp)))
        self.weight_vector = OR

        # counter = 0
        # self.weight_dict={}
        # for wt in self.weight_vector:
        #     self.weight_dict[vocab_list[counter]] = wt
        #     counter += 1

        # reverse_weight_list = self.weight_dict.keys()
        # reverse_weight_list.sort(self.weight_sort_fun)

        # OR_100 =  reverse_weight_list[:meaning_words] + reverse_weight_list[-1*meaning_words:]

        # self.most_senti_words = [w for w in nbsvm_100 if w in OR_100 and w in wfo_100]

        # for w in self.most_senti_words:
        #     self.weight_dict[w] = self.weight_dict[w] * 10

        # print len(self.most_senti_words)

        # print self.most_senti_words


    def compute_sws_w2v_feature_vecs(self, rev, w2v):
        feature_vec = np.zeros((w2v.dimension,), dtype="float32")
        nwords = 0.
        for word in rev.split():
            if word in w2v.vocab:
                try:
                    sws_w2v = w2v[word] #* self.weight_vector[self.reverse_vocab_dict[word]]         #self.reverse_vocab_dict[word]  vocab_list.index(word)
                    feature_vec = np.add(feature_vec, sws_w2v)
                    nwords += 1
                except KeyError:
                    pass

        if nwords==0.:
            print "Unkown test rev"
            feature_vec = np.random.uniform(-0.25,0.25, w2v.dimension)

        # binary_weight_vector = binary_vector * self.weight_vector
        # feature_vec = binary_weight_vector.dot(self.train_vocab_vectors)

        feature_vec = unit_vec(feature_vec)   # Normalization
        return feature_vec

    def get_sws_w2v_feature_vecs(self, w2v_model):
        """
            Given a set of documents calculate the average feature vector for each one
        """
        print "computing supervised weight vectors..."
        self.compute_w2v_weight_vector()
        self.get_train_vocab_list()        
        # num_vocab = len(self.train_vocab)
        # vocab_list = self.get_train_vocab_list()
        # counter = 0.
        # vacob_array = np.zeros((num_vocab, w2v_model.dimension), dtype="float32")
        # for w in vocab_list:
        #     if w in w2v_model.vocab:
        #         vacob_array[counter] = w2v_model[w]
        #     counter = counter + 1.
        # self.train_vocab_vectors = vacob_array

        print "computing train_doc_vecs..."
        train_doc_vecs = np.zeros((len(self.train_data), w2v_model.dimension), dtype="float32")
        counter = 0.
        for rev in self.train_data:
            train_doc_vecs[counter] = self.compute_sws_w2v_feature_vecs(rev, w2v_model)
            counter = counter + 1.
            # print counter

        print "computing test_doc_vecs..."
        test_doc_vecs = np.zeros((len(self.test_data), w2v_model.dimension), dtype="float32")
        counter = 0.
        for rev in self.test_data:
            test_doc_vecs[counter] = self.compute_sws_w2v_feature_vecs(rev, w2v_model)
            counter = counter + 1.

        self.train_doc_vecs = train_doc_vecs
        self.test_doc_vecs = test_doc_vecs
#############################################



######## word vecs centroid bags ###########
    def cluster_cre_adjust_tf_idf(self, w2v):
        word_centroid_map = w2v.word_centroid_map
        num_centroids = max(word_centroid_map.values()) + 1
        cluster_vocab = defaultdict(float)
        cluster_neg_counts = defaultdict(float)
        cluster_pos_counts = defaultdict(float)
        for word in self.train_vocab:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                cluster_vocab[index] += self.train_vocab[word]
                if word in self.pos_counts:
                    cluster_pos_counts[index] += self.pos_counts[word]
                if word in self.neg_counts:
                    cluster_neg_counts[index] += self.neg_counts[word]

        gamma = 0.8
        s_hat_average = 0.0
        cre_adjust_weight= np.zeros(num_centroids, dtype="float32")
        for w in cluster_vocab:
            s_hat = ((cluster_neg_counts[w])**2 +(cluster_pos_counts[w])**2) / (cluster_vocab[w])**2
            s_hat_average += (cluster_vocab[w] * s_hat) / self.train_word_nums

        for w in cluster_vocab:
            s_bar = ((cluster_neg_counts[w])**2 +(cluster_pos_counts[w])**2 + s_hat_average * gamma) / ((cluster_vocab[w])**2 + gamma)
            cre_adjust_weight[w] = 0.5 + s_bar

        zero_clusters = np.where(cre_adjust_weight == 0.0)[0]
        print "%d zero_clusters" % len(zero_clusters)
        for ix in zero_clusters:
            clu = "Cluster " + str(ix)
            words = w2v.clusters_dict[clu]
            #print words
            
            sim_words = w2v.most_similar(words[0])
            # print 'similar words:', sim_words
            for sim_word in sim_words:
                cluster = w2v.word_centroid_map[sim_word]
                if cluster not in zero_clusters:
                    cre_adjust_weight[ix] = cre_adjust_weight[cluster]
                    break
                else:
                    continue
        return cre_adjust_weight

    def create_bag_of_centroids(self, w2v, cre_adjust=False ):
        """
            Define a function to create bags of centroids
        """
        #
        # The number of clusters is equal to the highest cluster index
        # in the word / centroid map
        word_centroid_map = w2v.word_centroid_map
        num_centroids = max(word_centroid_map.values()) + 1
        train_centroids = np.zeros((len(self.train_data), num_centroids), dtype="float32" )
        test_centroids = np.zeros((len(self.test_data), num_centroids), dtype="float32" )

        counter = 0.
        for rev in self.train_data:
            word_list = rev.split()
            # Pre-allocate the bag of centroids vector (for speed)
            bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
            # Loop over the words in the review. If the word is in the vocabulary,
            # find which cluster it belongs to, and increment that cluster count
            # by one
            for word in word_list:
                if word in word_centroid_map:
                    index = word_centroid_map[word]
                    bag_of_centroids[index] += 1
                else:
                    pass
                    # print "unknown word", word
            train_centroids[counter] = bag_of_centroids
            counter += 1
        
        if cre_adjust:
            cluster_cre_adjust_weight = self.cluster_cre_adjust_tf_idf(w2v)
            train_centroids = train_centroids * cluster_cre_adjust_weight
        
        transformer = TfidfTransformer(sublinear_tf=cre_adjust)
        train_tfidf_centroids = transformer.fit_transform(train_centroids)
        self.train_doc_vecs = train_tfidf_centroids

        counter = 0.
        for rev in self.test_data:
            word_list = rev.split()
            # Pre-allocate the bag of centroids vector (for speed)
            bag_of_centroids = np.zeros(num_centroids, dtype="float32" )
            # Loop over the words in the review. If the word is in the vocabulary,
            # find which cluster it belongs to, and increment that cluster count
            # by one
            for word in word_list:
                if word in word_centroid_map:
                    index = word_centroid_map[word]
                    bag_of_centroids[index] += 1
                else:
                    pass
                    # print "unknown word", word
            test_centroids[counter] = bag_of_centroids
            counter += 1

        if cre_adjust:
            test_centroids = test_centroids * cluster_cre_adjust_weight
        test_tfidf_centroids = transformer.transform(test_centroids)
        self.test_doc_vecs = test_tfidf_centroids
############################################



######## word vecs tf-idf scheme ###########
    def build_w2v_tf_matrix(self, revs, tf_data):
        doc_num = len(revs)
        # num_vocab = len(self.vocab)
        # vocab_list = self.get_vocab_list()
        num_vocab = len(self.train_vocab)
        vocab_list = self.get_train_vocab_list()
        tf_matrix = np.zeros((doc_num, num_vocab), dtype="float32")

        counter = 0
        for rev in revs:
            tf = tf_data[counter]
            tf_list = np.zeros(num_vocab, dtype="float32")
            for w in tf:
                try:
                    i = vocab_list.index(w)
                    tf_list[i] = tf[w]
                except ValueError:
                    # Ignore out-of-vocabulary items
                    continue
            if np.array_equal(tf_list, np.zeros(num_vocab, dtype="float32")):
                print "Zero tf array for test doc"
                print rev
                #tf_list = np.random.uniform(0, 2, num_vocab)
                #print tf_list
            
            # # normalize tf list
            # tf_norm_list = 0.5 + 0.5 * tf_list / np.amax(tf_list)
            # print tf_norm_list
            # tf_matrix[counter] = tf_norm_list

            tf_matrix[counter] = tf_list
            counter = counter + 1

        return tf_matrix

    def cre_adjust_w2v_tf_idf(self, w2v):
        """
            cre-tfidf weight based on w2v (with all vocab)
        """
        gamma = 0.8
        vocab_list = self.get_train_vocab_list()
        s_hat_average = 0.0
        cre_adjust_weight= np.zeros(len(self.train_vocab), dtype="float32")
        for w in vocab_list:
            if w in self.train_vocab:
                s_hat = ((self.neg_counts[w])**2 +(self.pos_counts[w])**2) / (self.train_vocab[w])**2
                s_hat_average += (self.train_vocab[w] * s_hat) / self.train_word_nums

        for w in vocab_list:
            if w in self.train_vocab:
                s_bar = ((self.neg_counts[w])**2 +(self.pos_counts[w])**2 + s_hat_average * gamma) / ((self.train_vocab[w])**2 + gamma)
                i = vocab_list.index(w)
                cre_adjust_weight[i] = 0.5 + s_bar

        # # for words in vocabulary but not in train_vocab
        # zero_weights = np.where(cre_adjust_weight == 0.0)[0]
        # print "%d unknown words" % len(zero_weights)  # 936 words
        # for ix in zero_weights:
        #     word = vocab_list[ix]
        #     # print "unknown word:", word
        #     sim_words = w2v.most_similar(word)
        #     #print 'similar words:', sim_words
        #     for sim_word in sim_words:
        #         if sim_word in self.train_vocab:
        #             cre_adjust_weight[ix] = cre_adjust_weight[vocab_list.index(sim_word)]
        #            # print "substitution word:", sim_word
        #             #print "**********************************"
        #             break
        #print cre_adjust_weight
        # print "unknown word:", word 
        # print 'similar words:', sim_words
        # print sim_word
        return cre_adjust_weight

    def compute_tf_idf_feature_vecs(self,w2v, tfidf_matrix):
        num_doc = tfidf_matrix.shape[0]
        num_vocab = len(self.train_vocab)
        vocab_list = self.get_train_vocab_list()
        # num_vocab = len(self.vocab)
        # vocab_list = self.get_vocab_list()
        counter = 0.
        vacob_array = np.zeros((num_vocab, w2v.dimension), dtype="float32")
        unit_doc_vecs = np.zeros((num_doc, w2v.dimension), dtype="float32")
        for w in vocab_list:
            if w in w2v.vocab:
                vacob_array[counter] = w2v[w]
            counter = counter + 1.
        tf_idf_doc_vecs = tfidf_matrix.dot(vacob_array)

        # normalize the vecs
        counter = 0.
        for vec in tf_idf_doc_vecs:
            unit_doc_vecs[counter] = unit_vec(vec)
            counter += 1

        return unit_doc_vecs

    def get_tf_idf_feature_vecs(self, w2v, cre_adjust=False):
        """
            tf-idf weight scheme
        """
        #print " Computing train tf matrix..."
        train_tf_matrix = self.build_w2v_tf_matrix(self.train_data, self.train_tf)
        if cre_adjust:
            cre_adjust_weight = self.cre_adjust_w2v_tf_idf(w2v)
            train_tf_matrix = train_tf_matrix * cre_adjust_weight
        transformer = TfidfTransformer(sublinear_tf=cre_adjust)
        train_tfidf_matrix = transformer.fit_transform(train_tf_matrix)
        #print " Computing train tf_idf_doc_vecs..."
        self.train_doc_vecs = self.compute_tf_idf_feature_vecs(w2v,train_tfidf_matrix)
        # print 'train_doc_vecs'
        # print self.train_doc_vecs[1]
        # print "unit_vec"
        # print unit_vec(self.train_doc_vecs[1])

        #print " Computing test tf matrix..."
        test_tf_matrix = self.build_w2v_tf_matrix(self.test_data, self.test_tf)
        if cre_adjust:
            test_tf_matrix = test_tf_matrix * cre_adjust_weight
        #print " Computing test tf_idf_doc_vecs..."
        test_tfidf_matrix = transformer.transform(test_tf_matrix)
        self.test_doc_vecs = self.compute_tf_idf_feature_vecs(w2v, test_tfidf_matrix)
#############################################



######## word vecs average scheme ##########
    def compute_feature_vec(self, rev, w2v):
        feature_vec = np.zeros((w2v.dimension,), dtype="float32")
        nwords = 0.
        for word in rev.split():
            if word in w2v.vocab:
                nwords = nwords + 1.
                feature_vec = np.add(feature_vec, w2v[word])

        if nwords == 0:
            print "unkown words, something wrong !!!"
            feature_vec = np.random.uniform(-0.25,0.25, w2v.dimension)
        else:
            feature_vec = np.divide(feature_vec, nwords)
        avg_feature_vec = unit_vec(feature_vec)   # Normalization
        return avg_feature_vec

    def get_avg_feature_vecs(self, w2v_model):
        """
            Given a set of documents calculate the average feature vector for each one
        """
        train_doc_vecs = np.zeros((len(self.train_data), w2v_model.dimension), dtype="float32")
        counter = 0.
        for rev in self.train_data:
            train_doc_vecs[counter] = self.compute_feature_vec(rev, w2v_model)
            counter = counter + 1.

        test_doc_vecs = np.zeros((len(self.test_data), w2v_model.dimension), dtype="float32")
        counter = 0.
        for rev in self.test_data:
            test_doc_vecs[counter] = self.compute_feature_vec(rev, w2v_model)
            counter = counter + 1.

        self.train_doc_vecs = train_doc_vecs
        self.test_doc_vecs = test_doc_vecs
############################################



### Supervised Weight Scheme binary and w2v #####
    def weight_sort_fun(self,x,y):
        return cmp(self.weight_dict[y], self.weight_dict[x])

    def compute_sws_weight(self, alpha=1):
        vocab_list = self.get_train_vocab_list()
        d = len(vocab_list)
        Fp, Fn = np.ones(d) * alpha , np.ones(d) * alpha
        for w in vocab_list:
            Fp[vocab_list.index(w)] += self.pos_counts[w]
            Fn[vocab_list.index(w)] += self.neg_counts[w]
        Fw = Fp + Fn
        Fp /= abs(Fp).sum()
        Fn /= abs(Fn).sum()

        meaning_words = int(d/2) - 500

        # #### nbsvm ###
        # nbsvm_ratio = np.log(Fp/Fn)
        # self.weight_vector = nbsvm_ratio

        # counter = 0
        # self.weight_dict={}
        # for wt in self.weight_vector:
        #     self.weight_dict[vocab_list[counter]] = wt
        #     counter += 1

        # reverse_weight_list = self.weight_dict.keys()
        # reverse_weight_list.sort(self.weight_sort_fun)

        # nbsvm_100 =  reverse_weight_list[:meaning_words] + reverse_weight_list[-1*meaning_words:]
        # self.most_senti_words = nbsvm_100



        # ######## WFO #######
        # lam = 0.01  # 0.1
        # wfo = Fp**lam * np.log( (Fp/Fn)**(1-lam))
        # self.weight_vector = wfo

        # counter = 0
        # self.weight_dict={}
        # for wt in self.weight_vector:
        #     self.weight_dict[vocab_list[counter]] = wt
        #     counter += 1

        # reverse_weight_list = self.weight_dict.keys()
        # reverse_weight_list.sort(self.weight_sort_fun)

        # wfo_100 =  reverse_weight_list[:meaning_words] + reverse_weight_list[-1*meaning_words:]
        # self.most_senti_words = wfo_100

        # #### odds rotio #####

        OR = np.log(Fp*(1-Fn)/(Fn*(1-Fp)))
        self.weight_vector = OR

        counter = 0
        self.weight_dict={}
        for wt in self.weight_vector:
            self.weight_dict[vocab_list[counter]] = wt
            counter += 1

        reverse_weight_list = self.weight_dict.keys()
        reverse_weight_list.sort(self.weight_sort_fun)

        OR_100 =  reverse_weight_list[:meaning_words] + reverse_weight_list[-1*meaning_words:]
        self.most_senti_words = OR_100

        # self.most_senti_words = [w for w in nbsvm_100 if w in OR_100 and w in wfo_100]

        senti_weight_vector = np.zeros((len(self.most_senti_words),), dtype="float32")
        counter = 0.
        for w in self.most_senti_words:
            index = vocab_list.index(w)
            senti_weight_vector[counter] = self.weight_vector[index]
            counter += 1

        self.senti_weight_vector = senti_weight_vector

        # print len(self.most_senti_words)

        # print self.most_senti_words


    def sws_w2v_vectors(self, w2v=None):
        self.count_data()
        self.compute_sws_weight()
        output = []
        vocab_list = self.most_senti_words
        weight_vector = self.senti_weight_vector
        # print vocab_list
        # print weight_vector
        # vocab_list = self.get_train_vocab_list()
        # weight_vector = self.weight_vector

        print len(vocab_list)
        # print vocab_list
        # print self.weight_vector

        for beg_line, tf_dict in zip(self.train_labels, self.train_tf):
            indexes = []
            for w in tf_dict:
                try:
                    indexes += [vocab_list.index(w)]
                    # indexes += [vocab_list.index(w)+len(vocab_list)]
                except ValueError:
                    pass
                    # sim_words = w2v.most_similar(w)
                    # for sim_word in sim_words:
                    #     if sim_word[0] in vocab_list and sim_word[1] > 0.9:  #
                    #         # print w, sim_word
                    #         indexes += [ vocab_list.index(sim_word[0]) ]
                    #         break
                    #     else:
                    #         continue

            indexes = list(set(indexes))
            indexes.sort()
            line = [str(beg_line)]
            for i in indexes:
                line += ["%i:%f" % (i + 1, weight_vector[i])]
            output += [" ".join(line)]

        output = "\n".join(output)
        with open("train-swsvecs.txt", "w") as f:
            f.writelines(output)

        output = []
        for beg_line, tf_dict in zip(self.test_labels, self.test_tf):
            indexes = []
            for w in tf_dict:
                try:
                    indexes += [vocab_list.index(w)]
                    # indexes += [vocab_list.index(w)+len(vocab_list)]
                except ValueError:
                    pass
                    # sim_words = w2v.most_similar(w)
                    # for sim_word in sim_words:
                    #     if sim_word[0] in vocab_list and sim_word[1] > 0.9:  #
                    #         # print w, sim_word
                    #         indexes += [ vocab_list.index( sim_word[0]) ]
                    #         break
                    #     else:
                    #         continue

            indexes = list(set(indexes))
            indexes.sort()
            line = [str(beg_line)]
            for i in indexes:
                line += ["%i:%f" % (i + 1, weight_vector[i])]
            output += [" ".join(line)]

        output = "\n".join(output)
        with open("test-swsvecs.txt", "w") as f:
            f.writelines(output)

        liblinear='liblinear-1.96'
        trainsvm = os.path.join(liblinear, "train")
        predictsvm = os.path.join(liblinear, "predict")
        os.system(trainsvm + " -s 0 train-swsvecs.txt model.logreg")
        os.system(predictsvm + " -b 1 test-swsvecs.txt model.logreg " + './outputs/SWS-TEST')
        # os.system(trainsvm + " train-swsvecs.txt model.logreg")
        # os.system(predictsvm + " test-swsvecs.txt model.logreg " + './outputs/SWS-TEST')
        os.remove("model.logreg")
        os.remove("train-swsvecs.txt")
        os.remove("test-swsvecs.txt")

    def sws_w2v_art_fun(self, sws='OR', ngram='1', w2v=None):
        ptrain_lines = [rev for rev in self.train_data if self.train_labels[self.train_data.index(rev)] == 1]
        ntrain_lines = [rev for rev in self.train_data if self.train_labels[self.train_data.index(rev)] == 0]
        ptest_lines = [rev for rev in self.test_data if self.test_labels[self.test_data.index(rev)] == 1]
        ntest_lines = [rev for rev in self.test_data if self.test_labels[self.test_data.index(rev)] == 0]
        ptrain_lines = "\n".join(ptrain_lines)
        ntrain_lines = "\n".join(ntrain_lines)
        ptest_lines = "\n".join(ptest_lines)
        ntest_lines = "\n".join(ntest_lines)

        with open ('./outputs/ptrain','w') as f:
            f.writelines(ptrain_lines)
        with open ('./outputs/ntrain','w') as f:
            f.writelines(ntrain_lines)
        with open ('./outputs/ptest','w') as f:
            f.writelines(ptest_lines)
        with open ('./outputs/ntest','w') as f:
            f.writelines(ntest_lines)

        if w2v==None:
            output = './outputs/SWS-TEST'
        else:
            output = './outputs/SWS-W2V-TEST'

        nbsvm.main('./outputs/ptrain', './outputs/ntrain', './outputs/ptest', './outputs/ntest', output, 'liblinear-1.96', ngram, sws, w2v)
#################################################



############# Load dataset functions ###########
    @classmethod
    def load_data(cls, data_folder, clean_string=True, vocab_name='vocab_name', save_vocab= False):
        """
        Load data with revs and labels, then clean it
        """
        orig_revs = []
        labels = []
        cv_split = []
        pos_file = data_folder[0]
        neg_file = data_folder[1]
        vocab = defaultdict(float)     # dict,if key不存在,返回默认值 0.0

        # stemmer = PorterStemmer()
        
        with open(pos_file, "rb") as f:
            for line in f:
                raw_rev = line.strip()
                if clean_string:
                    cleaned_rev = clean_str(raw_rev)  # return string
                else:
                    cleaned_rev = raw_rev.lower()
                # cleaned_rev = stemmer.stem(cleaned_rev)

                
                words = cleaned_rev.split()

                # words = [word for word in words if len(word) > 1]   # 去掉一个字的单词
                # stops = set(stopwords.words("english"))
                # words = [w for w in words if not w in stops]

                for w in words:
                    vocab[w] += 1     # vocab[word] = vocab[word] + 1  统计词频
                orig_rev = ' '.join(words)

                orig_revs.append(orig_rev)
                labels.append(1)
                cv_split.append(np.random.randint(0, 10))

        with open(neg_file, "rb") as f:
            for line in f:
                raw_rev = line.strip()
                if clean_string:
                    cleaned_rev = clean_str(raw_rev)
                else:
                    cleaned_rev = raw_rev.lower()
                # cleaned_rev = stemmer.stem(cleaned_rev)

                words = cleaned_rev.split()
                # words = [word for word in words if len(word) > 1]

                # stops = set(stopwords.words("english"))
                # words = [w for w in words if not w in stops]

                for w in words:
                    vocab[w] += 1
                
                orig_rev = ' '.join(words)
                # print orig_rev
                labels.append(0)
                orig_revs.append(orig_rev)
                cv_split.append(np.random.randint(0, 10))
                # print orig_rev

        # delete word if count=1
        # cleaned_revs = []
        # rare_words = []
        # for word in vocab:
            # if vocab[word] < 1:
                # rare_words.append(word)
        # for word in rare_words:
            # vocab.pop(word)
        # for rev in orig_revs:
            # cleaned_rev = [w for w in rev if w not in rare_words]
            # cleaned_revs.append( ' '.join(cleaned_rev) )

        # save vocabulary
        if save_vocab:
            with open(vocab_name,'wb') as f:
                for word in vocab:
                    str_vocab = word + ':' + str(vocab[word]) + "\n"
                    f.write(str_vocab)

        logging.debug("For function load_data() **********" )
        logging.debug("First rev: [%s]" % orig_revs[0])
        logging.debug("First label: %d" % labels[0])

        print " Total loaded revs: %d" % len(orig_revs)
        print " Total vocab size: %d" % len(vocab)

        print " First 50 revs:", orig_revs[:50]
        return cls(revs=orig_revs, labels=labels, cv_split=cv_split, vocab=vocab)  # 18350

    @classmethod
    def load_splited_data(cls, data_folder, clean_string=True, vocab_name='vocab_name', save_vocab= False):
        """
        Load data with revs and labels, then clean it
        """
        train_pos_file = data_folder[0]
        train_neg_file = data_folder[1]
        test_pos_file = data_folder[2]
        test_neg_file = data_folder[3]

        vocab = defaultdict(float)     # dict,if key不存在,返回默认值 0.0

        result = DocVectors(revs=[], labels=[], cv_split=[], vocab=[])  # 18350
        result.train_data = []
        result.train_labels =[]
        result.test_data = []
        result.test_labels =[]


        with open(train_pos_file, "rb") as f:
            for line in f:
                raw_rev = line.strip()
                if clean_string:
                    cleaned_rev = clean_str(raw_rev)  # return string
                else:
                    cleaned_rev = raw_rev.lower()
                # cleaned_rev = stemmer.stem(cleaned_rev)

                words = cleaned_rev.split()
                # words = [word for word in words if len(word) > 1]   # 去掉一个字的单词
                # stops = set(stopwords.words("english"))
                # words = [w for w in words if not w in stops]

                for w in words:
                    vocab[w] += 1     # vocab[word] = vocab[word] + 1  统计词频
                orig_rev = ' '.join(words)

                result.train_data.append(orig_rev)
                result.train_labels.append(1)

        with open(train_neg_file, "rb") as f:
            for line in f:
                raw_rev = line.strip()
                if clean_string:
                    cleaned_rev = clean_str(raw_rev)
                else:
                    cleaned_rev = raw_rev.lower()
                # cleaned_rev = stemmer.stem(cleaned_rev)

                words = cleaned_rev.split()
                # words = [word for word in words if len(word) > 1]

                # stops = set(stopwords.words("english"))
                # words = [w for w in words if not w in stops]

                for w in words:
                    vocab[w] += 1

                orig_rev = ' '.join(words)
                result.train_data.append(orig_rev)
                result.train_labels.append(0)

        with open(test_pos_file, "rb") as f:
            for line in f:
                raw_rev = line.strip()
                if clean_string:
                    cleaned_rev = clean_str(raw_rev)  # return string
                else:
                    cleaned_rev = raw_rev.lower()
                # cleaned_rev = stemmer.stem(cleaned_rev)

                words = cleaned_rev.split()
                # words = [word for word in words if len(word) > 1]   # 去掉一个字的单词
                # stops = set(stopwords.words("english"))
                # words = [w for w in words if not w in stops]

                for w in words:
                    vocab[w] += 1     # vocab[word] = vocab[word] + 1  统计词频
                orig_rev = ' '.join(words)
                result.test_data.append(orig_rev)
                result.test_labels.append(1)

        with open(test_neg_file, "rb") as f:
            for line in f:
                raw_rev = line.strip()
                if clean_string:
                    cleaned_rev = clean_str(raw_rev)
                else:
                    cleaned_rev = raw_rev.lower()
                # cleaned_rev = stemmer.stem(cleaned_rev)

                words = cleaned_rev.split()
                # words = [word for word in words if len(word) > 1]

                # stops = set(stopwords.words("english"))
                # words = [w for w in words if not w in stops]

                for w in words:
                    vocab[w] += 1

                orig_rev = ' '.join(words)
                result.test_data.append(orig_rev)
                result.test_labels.append(0)

        result.vocab = vocab
        # save vocabulary
        if save_vocab:
            with open(vocab_name,'wb') as f:
                for word in vocab:
                    str_vocab = word + ':' + str(vocab[word]) + "\n"
                    f.write(str_vocab)
        
        print " Total loaded train revs: %d" % len(result.train_data)  
        print " Total loaded test revs: %d" % len(result.test_data)  
        print " Total vocab size: %d" % len(vocab)

        print result.train_data[:5] 
        return result

    def train_test_split(self, cv):
        """
        split the data into train_data and test_data
        """
        """
        :return:
        """
        self.train_data = []
        self.train_labels =[]
        self.test_data = []
        self.test_labels =[]

        counter = 0
        for rev in self.revs:
            if self.cv_split[counter] == cv:
                self.test_data.append(rev)
                self.test_labels.append(self.labels[counter])
            else:
                self.train_data.append(rev)
                self.train_labels.append(self.labels[counter])
            counter += 1

        # print " Size of train set: %d" % len(self.train_data)
        # print " Size of data set: %d" % len(self.test_data)

        logging.debug("For function train_test_split() **********")
        logging.debug("First test rev: [%s]" % self.test_data[0])
        logging.debug("First test label: %d" % self.test_labels[0])

    def count_data(self, save_vocab=False, vocab_name="vocabulary"):
        """
        traverse data to build vocab, train_word_nums, tf lists, and pos_counts
        """
        train_word_nums = 0.0
        pos_words_num = 0.0
        neg_words_num= 0.0
        train_vocab = defaultdict(float)
        pos_counts = defaultdict(float)  # for credibility Adjustment tf
        neg_counts = defaultdict(float)
        train_tf = []
        test_tf = []
        index = 0
        for rev in self.train_data:
            words = rev.split()
            tf = defaultdict(float)  # Term Frequencies 统计每个doc内的单词词频
            label = self.train_labels[index]
            for word in words:
                train_word_nums += 1
                train_vocab[word] += 1
                tf[word] += 1
                if label:
                    pos_words_num += 1
                    pos_counts[word] += 1
                else:
                    neg_words_num += 1
                    neg_counts[word] += 1
            train_tf.append(tf)
            index += 1

        index = 0
        for rev in self.test_data:
            words = rev.split()
            tf = defaultdict(float)  # Term Frequencies 统计每个doc内的单词词频
            label = self.test_labels[index]
            for word in words:
                tf[word] += 1
                # train_word_nums += 1
                # if label:
                #     pos_counts[word] += 1
                # else:
                #     neg_counts[word] += 1
            test_tf.append(tf)
            index += 1

        self.train_word_nums = train_word_nums
        self.pos_words_num = pos_words_num
        self.neg_words_num = neg_words_num
        self.train_vocab = train_vocab
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def sort_fun(self,x,y):
        return cmp(self.vocab[y], self.vocab[x])

    def get_vocab_list(self):
        """
            get the word list based on reverse count
        """
        reverse_vocab_list = self.vocab.keys()
        reverse_vocab_list.sort(self.sort_fun)

        #print reverse_vocab_list
        # for word in reverse_vocab_list:
        #    print word, ':', self.vocab[word],' ',
        return reverse_vocab_list
#################################################



############# sklearn bag of words ###########
    def get_bag_of_words_sklearn(self):
        """
            tf-idf weight scheme in sklearn
        """
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
        self.train_doc_vecs = vectorizer.fit_transform(self.train_data)
        self.test_doc_vecs = vectorizer.transform(self.test_data)
#################################################



########### custom bag of words ############
    def train_sort_fun(self,x,y):
        return cmp(self.train_vocab[y], self.train_vocab[x])

    def get_train_vocab_list(self):
        """
            get the train word list based on reverse count
        """
        reverse_vocab_list = self.train_vocab.keys()
        reverse_vocab_list.sort(self.train_sort_fun)

        self.reverse_vocab_dict={}
        for i,word in enumerate(reverse_vocab_list):
            self.reverse_vocab_dict[word] = i

        return reverse_vocab_list

    def build_tf_matrix(self, revs, tf_data):
        """
        build_tf_matrix
        """
        doc_num = len(revs)
        num_vocab = len(self.train_vocab)
        tf_matrix = np.zeros((doc_num, num_vocab), dtype="float32")
        print " tf_matrix:(%d, %d)" % (doc_num,num_vocab)

        counter = 0
        vocab_list = self.get_train_vocab_list()
        for rev in revs:
            tf = tf_data[counter]
            tf_list = np.zeros(num_vocab, dtype="float32")
            for w in tf:
                try:
                    i = vocab_list.index(w)
                    tf_list[i] = tf[w]
                except ValueError:
                    continue
                    # # Ignore out-of-vocabulary items

            if np.array_equal(tf_list, np.zeros(num_vocab, dtype="float32")):
                print "Zero tf array for test doc"
                print rev
                #tf_list = np.random.uniform(0,2, num_vocab)
                #print tf_list

            tf_matrix[counter] = tf_list
            counter = counter + 1

        return tf_matrix

    def cre_adjust_tf_idf(self):
        # print " Creating cre_adjust_tf_idf weight"
        gamma = 0.8
        vocab_list = self.get_train_vocab_list()
        s_hat_average = 0.0
        cre_adjust_weight= np.zeros(len(self.train_vocab), dtype="float32")

        for w in vocab_list:
            Fp = self.pos_counts[w]
            Fn = self.neg_counts[w]
            s_hat = ( Fn **2 +Fp**2) / (Fn+ Fp)**2
            s_hat_average += (self.train_vocab[w] * s_hat) / self.train_word_nums

        for w in vocab_list:
            Fp = self.pos_counts[w]
            Fn = self.neg_counts[w]
            s_bar = (Fn **2 +Fp**2 + s_hat_average * gamma) / ((Fn+ Fp)**2+ gamma)
            i = vocab_list.index(w)
            cre_adjust_weight[i] = 0.5 + s_bar

        return cre_adjust_weight

    def get_bag_of_words(self, cre_adjust=False):
        """
            Bag of word with tf-idf weight scheme
        """
        train_tf_matrix = self.build_tf_matrix(self.train_data, self.train_tf)

        if cre_adjust:
            cre_adjust_weight = self.cre_adjust_tf_idf()
            train_tf_matrix = train_tf_matrix * cre_adjust_weight
        transformer = TfidfTransformer(sublinear_tf=cre_adjust)   # sublinear_tf=True, selected for cre_adjust = true
        train_tfidf_matrix = transformer.fit_transform(train_tf_matrix)
        self.train_doc_vecs = train_tfidf_matrix

        #print " Computing test idf tf matrix..."
        test_tf_matrix = self.build_tf_matrix( self.test_data, self.test_tf)
        if cre_adjust:
            test_tf_matrix = test_tf_matrix * cre_adjust_weight
        test_tfidf_matrix = transformer.transform(test_tf_matrix)
        self.test_doc_vecs = test_tfidf_matrix
############################################







