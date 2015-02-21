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
from nltk.corpus import stopwords
import logging
logging.basicConfig(level=logging.INFO)
# from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer   # count the occurrences words and build a matrix (doc X vocab)
from sklearn.feature_extraction.text import TfidfTransformer  # transfer occurrences into tf or tfidf
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.vocab = vocab
        self.train_vocab = {}
        self.train_tf = []
        self.test_tf = []
        self.pos_counts = {}
        self.neg_counts = {}
        self.clusters = clusters

    # def train_test_split(self, test_size=0.25):
    #     """
    #     split the data into train_data and test_data
    #     """
    #     """
    #     :return:
    #     """
    #     self.train_data = []
    #     self.train_labels =[]
    #     self.test_data = []
    #     self.test_labels =[]
    #     self.train_data, self.test_data, self.train_labels, self.test_labels = \
    #         train_test_split(self.revs, self.labels , test_size=test_size, random_state=42)

    #     logging.debug("For function train_test_split() **********")
    #     logging.debug("Train numbers %f" % len(self.train_data))
    #     logging.debug("First test rev: [%s]" % self.test_data[0])
    #     logging.debug("First test label: %d" % self.test_labels[0])

    def count_data(self, save_vocab=False, vocab_name="vocabulary"):
        """
        traverse data to build vocab, train_word_nums, tf lists, and pos_counts
        """
        train_word_nums = 0.0
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
                    pos_counts[word] += 1
                else:
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

        # # delete word if count=1
        # rare_words = []
        # for word in vocab:
        #     if vocab[word] < 2:
        #         rare_words.append(word)
        #         train_word_nums -= 1
        # for word in rare_words:
        #     vocab.pop(word)
        #     if word in train_vocab:
        #         train_vocab.pop(word)
        #     if word in pos_counts:
        #         pos_counts.pop(word)
        #     if word in neg_counts:
        #         neg_counts.pop(word)
        # for tf in train_tf:
        #     tf_rare_words=[]
        #     for word in tf:
        #         if word in rare_words:
        #             tf_rare_words.append(word)
        #     for word in tf_rare_words:
        #         tf.pop(word)

        self.train_word_nums = train_word_nums
        self.train_vocab = train_vocab
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

        # # save vocabulary
        # if save_vocab:
        #     with open(vocab_name,'wb') as f:
        #         reverse_vocab_list = self.get_vocab_list()
        #         for word in reverse_vocab_list:
        #             str_vocab = word + ':' + str(self.vocab[word]) + "\n"
        #             f.write(str_vocab)
        #             # print str_vocab,

        # logging.debug("Total num of train words: %f, size of vocab: %f" % (train_word_nums, len(vocab)))

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

########### sklearn bag of words ###########
    def get_bag_of_words_sklearn(self):
        """
            tf-idf weight scheme in sklearn
        """
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)  
        self.train_doc_vecs = vectorizer.fit_transform(self.train_data)
        self.test_doc_vecs = vectorizer.transform(self.test_data)
############################################

########### custom bag of words ############
    def train_sort_fun(self,x,y):
        return cmp(self.train_vocab[y], self.train_vocab[x])

    def get_train_vocab_list(self):
        """
            get the train word list based on reverse count
        """
        reverse_vocab_list = self.train_vocab.keys()
        reverse_vocab_list.sort(self.train_sort_fun)

        #print reverse_vocab_list
        # for word in reverse_vocab_list:
        #    print word, ':', self.vocab[word],' ',
        return reverse_vocab_list

    def build_tf_matrix(self, revs, tf_data):
        doc_num = len(revs)
        # num_vocab = len(self.vocab)
        num_vocab = len(self.train_vocab)
        tf_matrix = np.zeros((doc_num, num_vocab), dtype="float32")

        counter = 0
        # vocab_list = self.get_vocab_list()
        vocab_list = self.get_train_vocab_list()

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

    def cre_adjust_tf_idf(self):
        # print " Creating cre_adjust_tf_idf weight"
        gamma = 0.8
        vocab_list = self.get_train_vocab_list()
        s_hat_average = 0.0
        cre_adjust_weight= np.zeros(len(self.train_vocab), dtype="float32")
        for w in vocab_list:
            s_hat = ((self.neg_counts[w])**2 +(self.pos_counts[w])**2) / (self.train_vocab[w])**2
            s_hat_average += (self.train_vocab[w] * s_hat) / self.train_word_nums

        for w in vocab_list:
            s_bar = ((self.neg_counts[w])**2 +(self.pos_counts[w])**2 + s_hat_average * gamma) / ((self.train_vocab[w])**2 + gamma)
            i = vocab_list.index(w)
            cre_adjust_weight[i] = 0.5 + s_bar

        #print cre_adjust_weight
        return cre_adjust_weight

    def get_bag_of_words(self, cre_adjust=False):
        """
            tf-idf weight scheme
        """
       # print " Computing train idf tf matrix..."
        train_tf_matrix = self.build_tf_matrix(self.train_data, self.train_tf)
        # print train_tf_matrix[0]
        # print self.get_vocab_list()
        if cre_adjust:
            cre_adjust_weight = self.cre_adjust_tf_idf()
            train_tf_matrix = train_tf_matrix * cre_adjust_weight
        transformer = TfidfTransformer(sublinear_tf=cre_adjust) # sublinear_tf=True, selected for cre_adjust = true
        train_tfidf_matrix = transformer.fit_transform(train_tf_matrix)
        self.train_doc_vecs = train_tfidf_matrix

        #print " Computing test idf tf matrix..."
        test_tf_matrix = self.build_tf_matrix(self.test_data, self.test_tf)
        if cre_adjust:
            test_tf_matrix = test_tf_matrix * cre_adjust_weight
        test_tfidf_matrix = transformer.transform(test_tf_matrix)
        self.test_doc_vecs = test_tfidf_matrix
############################################

    def compute_feature_vec(self, rev, w2v):
        feature_vec = np.zeros((w2v.dimension,), dtype="float32")
        nwords = 0.
        for word in rev.split():
            if word in w2v.vocab:
                nwords = nwords + 1.
                feature_vec = np.add(feature_vec, w2v[word])
        # avg_feature_vec = np.divide(feature_vec, rev["num_words"])
        if nwords == 0:
            print "something wrong !!!"
            avg_feature_vec = np.random.uniform(-0.25,0.25, w2v.dimension)
            # return
        else:
            avg_feature_vec = np.divide(feature_vec, nwords)
        avg_feature_vec = unit_vec(avg_feature_vec)
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
            #
            # # Print a status message every 1000th document
            # if counter % 1000. == 0. or counter == num_doc:
            #   percents = counter/num_doc * 100.0
            #   sys.stdout.write("Document %d of %d: %d%s" % (counter, num_doc, percents, '%\r'))
            #   sys.stdout.flush()
            #   if counter == num_doc:
            #     sys.stdout.write('\n')

        test_doc_vecs = np.zeros((len(self.test_data), w2v_model.dimension), dtype="float32")
        counter = 0.
        for rev in self.test_data:
            test_doc_vecs[counter] = self.compute_feature_vec(rev, w2v_model)
            counter = counter + 1.

        self.train_doc_vecs = train_doc_vecs
        self.test_doc_vecs = test_doc_vecs
        # print self.train_doc_vecs[1]
        # print "unit_vec"
        # print unit_vec(self.train_doc_vecs[1])
        # print self.test_doc_vecs[1]


    def compute_tf_idf_feature_vecs(self,w2v, tfidf_matrix):
        num_vocab = len(self.vocab)
        num_doc = tfidf_matrix.shape[0]
        vocab_list = self.get_vocab_list()
        counter = 0.
        vacob_array = np.zeros((num_vocab, w2v.dimension), dtype="float32")
        unit_doc_vecs = np.zeros((num_doc, w2v.dimension), dtype="float32")
        for w in vocab_list:
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
        train_tf_matrix = self.build_tf_matrix(self.train_data, self.train_tf)
        if cre_adjust:
            cre_adjust_weight = self.cre_adjust_tf_idf()
            train_tf_matrix = train_tf_matrix * cre_adjust_weight
        transformer = TfidfTransformer(sublinear_tf=True)
        train_tfidf_matrix = transformer.fit_transform(train_tf_matrix)
        #print " Computing train tf_idf_doc_vecs..."
        self.train_doc_vecs = self.compute_tf_idf_feature_vecs(w2v,train_tfidf_matrix)
        # print 'train_doc_vecs'
        # print self.train_doc_vecs[1]
        # print "unit_vec"
        # print unit_vec(self.train_doc_vecs[1])

        #print " Computing test tf matrix..."
        test_tf_matrix = self.build_tf_matrix(self.test_data, self.test_tf)
        if cre_adjust:
            test_tf_matrix = test_tf_matrix * cre_adjust_weight
        #print " Computing test tf_idf_doc_vecs..."
        test_tfidf_matrix = transformer.transform(test_tf_matrix)
        self.test_doc_vecs = self.compute_tf_idf_feature_vecs(w2v, test_tfidf_matrix)

    def cluster_cre_adjust_tf_idf(self, word_centroid_map):
        cluster_vocab = defaultdict(float)
        cluster_neg_counts = defaultdict(float)
        cluster_pos_counts = defaultdict(float)
        for word in self.vocab:
            index = word_centroid_map[word]
            cluster_vocab[index] += self.vocab[word]
            if word in self.pos_counts:
                cluster_pos_counts[index] += self.pos_counts[word]
            if word in self.neg_counts:
                cluster_neg_counts[index] += self.neg_counts[word]

        gamma = 1
        s_hat_average = 0.0
        cre_adjust_weight= np.zeros(len(cluster_vocab), dtype="float32")
        for w in cluster_vocab:
            s_hat = ((cluster_neg_counts[w])**2 +(cluster_pos_counts[w])**2) / (cluster_vocab[w])**2
            s_hat_average += (cluster_vocab[w] * s_hat) / self.train_word_nums

        for w in cluster_vocab:
            s_bar = ((cluster_neg_counts[w])**2 +(cluster_pos_counts[w])**2 + s_hat_average * gamma) / ((cluster_vocab[w])**2 + gamma)
            cre_adjust_weight[w] = 0.5 + s_bar

        return cre_adjust_weight

    def create_bag_of_centroids(self, word_centroid_map, cre_adjust=False ):
        """
            Define a function to create bags of centroids
        """
        #
        # The number of clusters is equal to the highest cluster index
        # in the word / centroid map

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
            train_centroids[counter] = bag_of_centroids
            counter += 1
        if cre_adjust:
            cluster_cre_adjust_weight = self.cluster_cre_adjust_tf_idf(word_centroid_map)
            train_centroids = train_centroids * cluster_cre_adjust_weight
        transformer = TfidfTransformer(sublinear_tf=True)
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
            test_centroids[counter] = bag_of_centroids
            counter += 1
        if cre_adjust:
            test_centroids = test_centroids * cluster_cre_adjust_weight
        test_tfidf_centroids = transformer.transform(test_centroids)
        self.test_doc_vecs = test_tfidf_centroids







    def cre_sim_weight_2(self,w2v):
        # print " Creating cre_adjust_tf_idf weight"
        gamma = 1
        vocab_list = self.get_vocab_list()
        s_hat_average = 0.0
        cre_adjust_weight= np.zeros(len(self.vocab), dtype="float32")
        for w in vocab_list:
            if w in self.train_vocab:
                s_hat = ((self.neg_counts[w])**2 +(self.pos_counts[w])**2) / (self.vocab[w])**2
                s_hat_average += (self.vocab[w] * s_hat) / self.train_word_nums

        for w in vocab_list:
            if w in self.train_vocab:
                s_bar = ((self.neg_counts[w])**2 +(self.pos_counts[w])**2 + s_hat_average * gamma) / ((self.vocab[w])**2 + gamma)
                i = vocab_list.index(w)
                cre_adjust_weight[i] = 0.5 + s_bar

        zero_weights = np.where(cre_adjust_weight == 0.0)[0]
        for ix in zero_weights:
            word = vocab_list[ix]
            print "unknown word:", word
            sim_words = w2v.most_similar(w)
            print 'similar words:', sim_words
            for sim_word in sim_words:
                if sim_word in self.train_vocab:
                    cre_adjust_weight[ix] = cre_adjust_weight[vocab_list.index(sim_word)]
                    break
                    print "substitution word:", sim_word
                    print "**********************************"

        #print cre_adjust_weight
        return cre_adjust_weight

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

        gamma = 100
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
        zero_clusters = np.where(cre_sim_clu_weight == 0.0)[0]
        print "zero_clusters: ", zero_clusters

        vocab_list = self.get_vocab_list()
        for w in vocab_list:
            index = vocab_list.index(w)
            cluster = w2v.word_centroid_map[w]
            # if test word in unknown cluster
            if cluster in zero_clusters:
                print "unknown cluster test word:", w
                sim_words = w2v.most_similar(w)
                print 'similar words:', sim_words
                for sim_word in sim_words:
                    clu = w2v.word_centroid_map[sim_word]
                    if clu in zero_clusters:
                        continue
                    else:
                        cluster = clu
                        print "substitution word:", sim_word
                        print "**********************************"
                        break

            cre_sim_word_weight[index] = cre_sim_clu_weight[cluster]

        return cre_sim_word_weight

    def cre_sim_doc_vecs(self, w2v):
        train_tf_matrix = self.build_tf_matrix(self.train_data, self.train_tf)

        cre_sim_word_weight = self.cre_sim_weight(w2v)
        train_tf_matrix = train_tf_matrix * cre_sim_word_weight

        transformer = TfidfTransformer(sublinear_tf=True)
        train_tfidf_matrix = transformer.fit_transform(train_tf_matrix)

        #print " Computing train tf_idf_doc_vecs..."
        self.train_doc_vecs = self.compute_tf_idf_feature_vecs(w2v, train_tfidf_matrix)

        #print " Computing test tf matrix..."
        test_tf_matrix = self.build_tf_matrix(self.test_data, self.test_tf)
        test_tf_matrix = test_tf_matrix * cre_sim_word_weight
        #print " Computing test tf_idf_doc_vecs..."
        test_tfidf_matrix = transformer.transform(test_tf_matrix)
        self.test_doc_vecs = self.compute_tf_idf_feature_vecs(w2v, test_tfidf_matrix)

    @classmethod
    def build_data_cv_1(cls, data_folder, cv=10, clean_string=True):
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


    @classmethod
    def load_data(cls, data_folder, clean_string=True):
        """
        Load data with revs and labels, then clean it
        """
        revs = []
        labels = []
        cv_split = []
        pos_file = data_folder[0]
        neg_file = data_folder[1]
        vocab = defaultdict(float)  # dict,if key不存在,返回默认值 0.0

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
                for w in words:
                    vocab[w] += 1     # vocab[word] = vocab[word] + 1  统计词频

                orig_rev = ' '.join(words)
                revs.append(orig_rev)
                labels.append(1)
                cv_split.append(np.random.randint(0, 10))

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
                for w in words:
                    vocab[w] += 1     # vocab[word] = vocab[word] + 1  统计词频
                
                orig_rev = ' '.join(words)
                labels.append(0)
                revs.append(orig_rev)
                cv_split.append(np.random.randint(0, 10))

        logging.debug("For function load_data() **********" )
        print " Total loaded revs: %d" % len(revs)
        print " Total vocab size: %d" % len(vocab)
        logging.debug("First rev: [%s]" % revs[0])
        logging.debug("First label: %d" % labels[0])
        return cls(revs=revs, labels=labels, cv_split=cv_split, vocab=vocab)

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

        print " Size of train set: %d" % len(self.train_data)
        print " Size of data set: %d" % len(self.test_data)

        logging.debug("For function train_test_split() **********")
        logging.debug("First test rev: [%s]" % self.test_data[0])
        logging.debug("First test label: %d" % self.test_labels[0])



