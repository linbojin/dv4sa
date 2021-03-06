#!/usr/bin/env python
# coding: utf-8

"""
    io.py
    Author: Linbo
    Date: 15.03.2015
"""

from __future__ import unicode_literals
import doc2vec


def load_word_vec(fname, vocab, cluster=False, kind='auto'):
    '''
    Loads a word vectors file
    '''
    if kind == 'auto':
        if fname.endswith('.bin'):
            kind = 'bin'
        elif fname.endswith('.txt'):
            kind = 'txt'
        else:
            raise Exception('Could not identify kind')
    if kind == 'bin':
        return doc2vec.WordVectors.load_bin_vec(fname, vocab, cluster)
    elif kind == 'txt':
        pass
        #return doc2vec.WordVectors.from_text(fname, *args, **kwargs)
    elif kind == 'mmap':
        pass
        #return doc2vec.WordVectors.from_mmap(fname, *args, **kwargs)
    else:
        raise Exception('Unknown kind')

def load_docs(data_folder, clean_string=True, vocab_name='vocab_name', save_vocab=False, splited=False):
    if splited:
        return doc2vec.DocVectors.load_splited_data(data_folder, clean_string, vocab_name, save_vocab)
    else:
        return doc2vec.DocVectors.load_data(data_folder, clean_string, vocab_name, save_vocab)