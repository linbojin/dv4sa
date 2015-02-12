#!/usr/bin/env python
# coding=utf-8

# from __future__ import unicode_literals
import numpy as np
import sys
from sklearn.cluster import KMeans
import time
from doc2vec.utility import unit_vec
from six import string_types
from numpy import ndarray, argsort, array, dot, float32 as REAL

class WordVectors(object):

    def __init__(self, vectors=None, cluster=False):
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
        self.vocab = {}
        self.index2word  = []  # map from a word's matrix index (int) to word (string)
        #self.syn0 = syn0
        self.vectors = {}
        self.dimension = 300
        self.cluster = cluster
        self.word_centroid_map={}


    def __getitem__(self, word):
        """
        Model['word'] return vectors
        """
        return self.get_vector(word)

    def get_vector(self, word):
        """
        Returns the word vector for `word` in the vocabulary
        """
        if word in self.vectors:
            return self.vectors[word]
        else:
            print 'Word not in vocabulary'
            return

    def get_w2v_centroid(self, sname = "wv-clusters"):
        """
        Run Kmean on wordvecs
        """
        start = time.time() # Start time
        word_vectors = self.syn0
        num_clusters = word_vectors.shape[0] / 5
        #print len(self.vocab)
        #print "num_clusters", str(num_clusters)

        kmeans_clustering = KMeans( n_clusters = num_clusters )
        idx = kmeans_clustering.fit_predict( word_vectors )

        word_centroid_map = dict(zip( self.index2word, idx ))
        self.word_centroid_map = word_centroid_map
        # Get the end time and print how long the process took
        end = time.time()
        elapsed = end - start
        print "*** Time for KMeans:", elapsed, "seconds."

        # save vocabulary
        # print "Saving wordvec clusters"
        with open(sname, 'wb') as f:
            for cluster in xrange(0,num_clusters):
                #
                # Print the cluster number
                f.write("Cluster %d \n" % cluster)
                f.write("[")
                #
                # Find all of the words for that cluster number, and print them out
                words = []
                for word, value in word_centroid_map.iteritems():
                    if value == cluster:
                        f.write("%s, " % word)
                f.write("]")
                f.write("\n\n")

        # # print first 10 word clusters
        # for cluster in xrange(0,10):
        #     #
        #     # Print the cluster number
        #     print "\nCluster %d" % cluster
        #     #
        #     # Find all of the words for that cluster number, and print them out
        #     words = []
        #     for word, value in word_centroid_map.iteritems():
        #         if value == cluster:
        #             words.append(word)
        #     print words

    @classmethod
    def load_bin_vec(cls, fname, vocab, cluster = False):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        index = 0
        with open(fname, "rb") as f:
            header = f.readline()
            # layer1_size = 300
            vocab_size, layer1_size = map(int, header.split())

            result = WordVectors()
            result.syn0 = np.zeros((len(vocab), layer1_size), dtype='float32')

            binary_len = np.dtype('float32').itemsize * layer1_size
            counter = 0.
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                    vector = np.fromstring(f.read(binary_len), dtype='float32')    # 将str data转换成float32  不太理解
                    uni_vector = unit_vec(vector)
                    result.vectors[word] = uni_vector

                    result.index2word.append(word)
                    result.syn0[index] = uni_vector
                    index += 1
                    #word_vecs[word] = vector
                else:
                    f.read(binary_len)     # 不添加vocab之外的wordvec,节省空间

                counter += 1
                # if counter % 10000. == 0. or counter == vocab_size:
                #     sys.stdout.write("Progress %d%s" % (counter/vocab_size * 100.0, '%\r') )
                #     sys.stdout.flush()
                #     if counter == vocab_size:
                #         sys.stdout.write('\n')

        result.vocab = vocab
        result.cluster = cluster
        # add_unknown_words
        min_df = 1
        k=layer1_size
        result.dimension = k
        for word in vocab:
            if word not in result.vectors and vocab[word] >= min_df:
                rd_vector = np.random.uniform(-0.25,0.25,k)   # 0.25
                rd_vector =  unit_vec(rd_vector)
                result.vectors[word] = rd_vector
                result.index2word.append(word)
                result.syn0[index] = rd_vector
                index += 1

        return result


    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.
        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy` and
        `distance` scripts in the original word2vec implementation.
        Example::
          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]
        """

        if isinstance(positive,string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                                else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                                 else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0[self.index2word.index(word)])
                all_words.add(self.index2word.index(word))
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = unit_vec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]