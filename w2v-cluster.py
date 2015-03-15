#!/usr/bin/env python
# coding = utf-8
"""
    Build word vector clusters for datasets
    input:  dataset
            word vecs
    output: vec clusters(classes)
            centroid_map
"""

import doc2vec
try:
    import cPickle as pickle
except ImportError:
    import pickle

def build_clusters(data_folder, path, dataset):
    print "Building clusters on", dataset
    vocab_name = path + dataset + ".vocab"
    d2v_model = doc2vec.load_docs(data_folder, clean_string=True, vocab_name=vocab_name, save_vocab=True)

    w2v_file = './datasets/wordvecs/GoogleNews-vectors-negative300.bin'
    # w2v_file = './datasets/wordvecs/vectors.bin'
    w2v_model = doc2vec.load_word_vec(w2v_file, d2v_model.vocab, cluster=True)
    sname = path + dataset + ".clusters"
    w2v_model.get_w2v_centroid(sname=sname)

    centroid_map = path + dataset + "-centroid-map.p"
    with open(centroid_map, "wb") as f:
        pickle.dump([w2v_model.word_centroid_map], f)     # create a pickle object
    print

print "Creating word vector clusters for datasets"
path = './datasets/'

dataset = 'rt-polarity'
# dataset = 'test'
data_folder = [path+dataset+".pos", path+dataset+".neg"]
build_clusters(data_folder, path, dataset)
# dataset = 'subj'
# data_folder = [path+dataset+".subjective", path+dataset+".objective"]
# build_clusters(data_folder, path, dataset)
print "Done!"
