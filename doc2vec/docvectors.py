#!/usr/bin/env python
# coding=utf-8

""""
Bag_of_words
"""
import numpy as np
import re
from collections import defaultdict
import sys
from doc2vec.utility import *
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

class DocVectors(object):

    def __init__(self, vocab, revs, pos_counts, neg_counts, word_nums, clusters=None):
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
        self.vocab = vocab
        self.word_nums = word_nums
        self.revs = revs
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts
        self.clusters = clusters

    def compute_feature_vec(self, rev, w2v):
        feature_vec = np.zeros((w2v.dimension,), dtype="float32")
        nwords = 0.
        for word in rev["text"].split():
            if word in w2v.vocab:
                nwords = nwords + 1.
                feature_vec = np.add(feature_vec, w2v[word])
        # avg_feature_vec = np.divide(feature_vec, rev["num_words"])
        if nwords == 0:
            print "something wrong !!!"
            return
        else:
            avg_feature_vec = np.divide(feature_vec, nwords)
            avg_feature_vec = unit_vec(avg_feature_vec)
            return avg_feature_vec

    def get_avg_feature_vecs(self, w2v_model):
        """
            Given a set of documents (each one is a structure {y,text,num_words,split} ),
            calculate the average feature vector for each one
        """
        num_doc = len(self.revs)
        counter = 0.
        for rev in self.revs:
            rev['doc_vec'] = self.compute_feature_vec(rev, w2v_model)
            counter = counter + 1.
            #
            # # Print a status message every 1000th document
            # if counter % 1000. == 0. or counter == num_doc:
            #   percents = counter/num_doc * 100.0
            #   sys.stdout.write("Document %d of %d: %d%s" % (counter, num_doc, percents, '%\r'))
            #   sys.stdout.flush()
            #   if counter == num_doc:
            #     sys.stdout.write('\n')

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
        #     print word, ':', self.vocab[word],' ',
        return reverse_vocab_list

    def get_tf_idf_feature_vecs(self, w2v, cre_adjust=False, BagOfWords = False):
        """
            tf-idf weight scheme
        """
        print " Computing tf_matrix..."
        num_doc = len(self.revs)
        num_vocab = len(self.vocab)
        tf_matrix = np.zeros((num_doc,num_vocab), dtype="float32")

        counter = 0.
        vocab_list = self.get_vocab_list()
        for rev in self.revs:
            tf = rev["term_fre"]
            tf_list = np.zeros(num_vocab, dtype="float32")
            for w in tf:
                i = vocab_list.index(w)
                tf_list[i] = tf[w]
            tf_matrix[counter] = tf_list
            counter = counter + 1.

        if cre_adjust:
            cre_adjust_weight = self.Cre_Adjust_tf_idf()
            tf_matrix = tf_matrix * cre_adjust_weight

        print " Computing tf idf sparse matrix..."
        #print "tf_matrix", tf_matrix
        transformer = TfidfTransformer(sublinear_tf=True)
        tfidf = transformer.fit_transform(tf_matrix)
        #print "tfidf",tfidf
        #print tfidf.toarray()

        if BagOfWords:
            tf_idf_doc_vecs = tfidf
        else:
            print " Computing tf_idf_doc_vecs..."
            counter = 0.
            vacob_array = np.zeros((num_vocab, w2v.dimension), dtype="float32")
            for w in vocab_list:
                vacob_array[counter] = w2v[w]
                counter = counter + 1.
            tf_idf_doc_vecs = tfidf.dot(vacob_array)
            # print "tf_idf_doc", tf_idf_doc_vecs[:2]

        counter = 0.
        for rev in self.revs:
            rev["doc_vec"]= unit_vec(tf_idf_doc_vecs[counter])    # normalize the vector
            #rev["doc_vec"]= tf_idf_doc_vecs[counter]    # normalize the vector
            counter = counter + 1.



    def Cre_Adjust_tf_idf(self):
        gamma = 1
        vocab_list = self.get_vocab_list()
        s_hat_average = 0.0
        cre_adjust_weight= np.zeros(len(self.vocab), dtype="float32")
        for w in vocab_list:
            s_hat = ((self.neg_counts[w])**2 +(self.pos_counts[w])**2) / (self.vocab[w])**2
            s_hat_average += (self.vocab[w] * s_hat) / self.word_nums

        for w in vocab_list:
            s_bar = ((self.neg_counts[w])**2 +(self.pos_counts[w])**2 + s_hat_average * gamma) / ((self.vocab[w])**2 + gamma)
            i = vocab_list.index(w)
            cre_adjust_weight[i] = (0.5 + s_bar)

        return cre_adjust_weight





    def create_bag_of_centroids( self, word_centroid_map ):
        """
            Define a function to create bags of centroids
        """
        #
        # The number of clusters is equal to the highest cluster index
        # in the word / centroid map

        num_centroids = max( word_centroid_map.values() ) + 1

        for rev in self.revs:
            word_list = rev["text"].slipt()

            # Pre-allocate the bag of centroids vector (for speed)
            bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
            # Loop over the words in the review. If the word is in the vocabulary,
            # find which cluster it belongs to, and increment that cluster count
            # by one
            for word in word_list:
                if word in word_centroid_map:
                    index = word_centroid_map[word]
                    bag_of_centroids[index] += 1

            rev["doc_vec"] = bag_of_centroids

    @classmethod
    def build_data_cv(cls, data_folder, cv=10, clean_string=True):
        """
        Loads data and split into 10 folds.
        """
        revs = []
        word_nums = 0.0
        pos_file = data_folder[0]
        neg_file = data_folder[1]
        vocab = defaultdict(float)  # dict,if key不存在,返回默认值 0.0
        pos_counts = defaultdict(float)  # for credibility Adjustment tf
        with open(pos_file, "rb") as f:
            for line in f:
                raw_rev = line.strip()
                if clean_string:
                    cleaned_rev = clean_str(raw_rev)
                else:
                    cleaned_rev = raw_rev.lower()
                    # type(cleaned_rev) is str

                words = cleaned_rev.split()
                words = [word for word in words if len(word) > 1]   # 去掉一个字的单词

                # stops = set(stopwords.words("english"))
                # words = [w for w in words if not w in stops]

                orig_rev = ' '.join(words)

                tf = defaultdict(float)  # Term Frequencies 统计每个doc内的单词词频
                for word in words:
                    vocab[word] += 1  # vocab[word] = vocab[word] + 1  统计词频
                    tf[word] += 1
                    pos_counts[word] += 1
                    word_nums += 1

                data_unit = {"y": 1,
                             "text": orig_rev,
                             "term_fre": tf,
                             "num_words": len(words),
                             "split": np.random.randint(0, cv)
                            }
                revs.append(data_unit)

        neg_counts = defaultdict(float)
        with open(neg_file, "rb") as f:
            for line in f:
                raw_rev = line.strip()
                if clean_string:
                    cleaned_rev = clean_str(raw_rev)
                else:
                    cleaned_rev = raw_rev.lower()

                words = cleaned_rev.split()
                words = [word for word in words if len(word) > 1]   # 去掉一个字的单词

                # stops = set(stopwords.words("english"))
                # words = [w for w in words if not w in stops]

                orig_rev = ' '.join(words)

                tf = defaultdict(float)  # Term Frequencies 统计每个doc内的单词词频
                for word in words:
                    vocab[word] += 1
                    tf[word] += 1
                    neg_counts[word] += 1
                    word_nums += 1

                doc_vec=[]
                data_unit = {"y": 0,
                             "text": orig_rev,
                             "term_fre": tf,
                             "num_words": len(words),
                             "split": np.random.randint(0, cv),
                             "doc_vec": doc_vec}

                revs.append(data_unit)

        return cls(vocab=vocab, revs=revs, pos_counts=pos_counts, neg_counts=neg_counts, word_nums=word_nums)



