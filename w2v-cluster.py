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

def build_clusters(dataset):
    print "Building clusters on", dataset
    data_folder = [path+dataset+".pos", path+dataset+".neg"]
    d2v_model = doc2vec.load_docs(data_folder, clean_string=True)
    d2v_model.train_test_split(test_size=0.1)
    name_vocab = path + dataset + "-vocab"
    d2v_model.count_data(name_vocab=name_vocab, save_vocab=True)

    w2v_file = './datasets/wordvecs/GoogleNews-vectors-negative300.bin'
    w2v_file = './datasets/wordvecs/vectors.bin'
    w2v_model = doc2vec.load_word_vec(w2v_file, d2v_model.vocab, cluster=True)
    clusters = path + dataset + "-clusters"
    w2v_model.get_w2v_centroid(sname=clusters)

    centroid_map = path + dataset + "-centroid-map.p"
    with open(centroid_map, "wb") as f:
        pickle.dump([w2v_model.word_centroid_map], f)     # create a pickle object
    print

print "Creating word vector clusters for datasets"
path = './datasets/'
datasets = ['rt-polarity', 'custrev', 'mpqa']
# datasets = ["test"]
for dataset in datasets:
    build_clusters(dataset)
print "Done!"
