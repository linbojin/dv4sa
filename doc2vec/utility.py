#!/usr/bin/env python
# coding=utf-8


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
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " ", string)
    # string = re.sub(r"n\'t", " ", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " ", string)
    string = re.sub(r"\'ll", " ", string)

    string = re.sub(r" \'\'", " ", string)   # delete ''xx''
    string = re.sub(r"\'\' ", " ", string)
    string = re.sub(r" \'", " ", string)     # delete 'xx'
    string = re.sub(r"\' ", " ", string)

    string = re.sub(r"\s{2,}", " ", string)     # 清除长空格,2格以上

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()