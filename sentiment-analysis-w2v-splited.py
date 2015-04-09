#!usr/bin/env python
# coding = utf-8

"""
    Sentiment Analysis based on word2vec for splieted data set
        Average
        tf-idf
        NBSVM
        OR
        WFO
    Author: Linbo
    Date: 15.03.2015
"""

import sys
import os
import numpy as np
import doc2vec

print "Sentiment Analysis based on Supervied Weight Scheme"
############ Load dataset ##################
print "Loading dataset ... "
path = './datasets/'
data_folder = [path+"aclImdb-train-pos.txt",path+"aclImdb-train-neg.txt",
               path+"aclImdb-test-pos.txt",path+"aclImdb-test-neg.txt"]

d2v_model = doc2vec.load_docs(data_folder, clean_string=False, splited=True)
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
print "Run Algorithms"
d2v_model.count_data()

d2v_model.get_avg_feature_vecs(w2v_model)        # 77.6 c=1  word vec average scheme
# d2v_model.get_tf_idf_feature_vecs(w2v_model)      # 77.3 c=1  word vec tf-idf scheme
# d2v_model.get_tf_idf_feature_vecs(w2v_model, cre_adjust=True)  # 76.9 c=1  word vec cre tf-idf scheme

# d2v_model.get_sws_w2v_feature_vecs(w2v_model, sws='NBSVM')
# d2v_model.get_sws_w2v_feature_vecs(w2v_model, sws='OR')
# d2v_model.get_sws_w2v_feature_vecs(w2v_model, sws='WFO', lamda = 0.1 )
# d2v_model.get_sws_w2v_feature_vecs(w2v_model, sws='WFO', lamda = 0.05 )
# d2v_model.get_sws_w2v_feature_vecs(w2v_model, sws='WFO', lamda = 0.01 )

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

sys.stdout.flush()
os.remove("accuracy")



























