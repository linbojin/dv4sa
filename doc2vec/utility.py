#!/usr/bin/env python
# coding: utf-8

"""
    utility.py
    Author: Linbo
    Date: 15.03.2015
"""

from numpy import linalg as LA
import numpy as np
import re

def unit_vec(vec):
    if np.all(vec == 0):
        print "all zero test"
        return vec
    else:
        return (1.0 / LA.norm(vec, ord=2)) * vec

def clean_str(str):
    string = str.strip('\'')  # delete \' at the beginning

    # string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)  # ****************  nbsvm 78.1%
    # string = re.sub(r"[^A-Za-z]", " ", string)  # no no no!!!!
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  # **************  nbsvm 78.2%    79.1
    string = re.sub(r"\'s", " s", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"can\'t", "can not", string)
    string = re.sub(r"won\'t", "will not", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " had", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r" \'\'", " ", string)   # delete ''xx''
    string = re.sub(r"\'\' ", " ", string)
    string = re.sub(r" \'", " ", string)     # delete 'xx'
    string = re.sub(r"\' ", " ", string)

    string = re.sub(r"\s{2,}", " ", string)    # 清除长空格,2格以上
    return string.strip().lower()

def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_".join(words[i:i+gram])]
    return tokens

