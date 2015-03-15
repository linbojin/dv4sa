import os
import pdb
import numpy as np
import argparse
from collections import Counter

def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += [" ".join(words[i:i+gram])]
    return tokens

def build_dict(f, grams):
    dic = Counter()
    for sentence in open(f).xreadlines():
        dic.update(tokenize(sentence, grams))
    return dic

def process_files(file_pos, file_neg, dic, r, outfn, grams, w2v):
    output = []
    for beg_line, f in zip(["1", "-1"], [file_pos, file_neg]):
        for l in open(f).xreadlines():
            tokens = tokenize(l, grams)
            indexes = []
            for t in tokens:
                try:
                    indexes += [dic[t]]
                except KeyError:
                    if w2v==None:
                        pass
                    else:
                        t = t.split()
                        try:
                            sim_words = w2v.most_similar(t)
                            for sim_word in sim_words:
                                if sim_word[0] in dic and sim_word[1] > 0.7:  #
                                    print t, sim_word
                            
                                    # if sim_word[1] > 0.8:
                                    #     print t, sim_word
                                    indexes += [dic[sim_word[0]]]
                                    break
                                else:
                                    continue
                        except KeyError:
                            pass

            indexes = list(set(indexes))
            indexes.sort()
            line = [beg_line]
            for i in indexes:
                line += ["%i:%f" % (i + 1, r[i])]
            output += [" ".join(line)]
    output = "\n".join(output)
    f = open(outfn, "w")
    f.writelines(output)
    f.close()

# def compute_ratio(poscounts, negcounts, alpha=1):
#     alltokens = list(set(poscounts.keys() + negcounts.keys()))
#     dic = dict((t, i) for i, t in enumerate(alltokens))
#     d = len(dic)
#     print "computing r..."
#     p, q = np.ones(d) * alpha , np.ones(d) * alpha
#     for t in alltokens:
#         p[dic[t]] += poscounts[t]
#         q[dic[t]] += negcounts[t]
#     p /= abs(p).sum()
#     q /= abs(q).sum()
#     r = np.log(p/q)
#     return dic, r


def compute_ratio(poscounts, negcounts, alpha=1, sws='NBSVM'):
    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    print "computing r..."
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    Fp = p
    Fn = q
    if sws=='NBSVM':
        print 'NBSVM'
        r = np.log(p/q)                # 88.612 91.56  91.82%         # sent05 78.2% 79.1        
    elif sws=='OR':
        print 'OR'
        r = np.log(p*(1-q)/(q*(1-p)))  # 88.604 91.56  91.87          # sent05 78.0% 79.0   78.24%
    elif sws=='WFO':
        print 'WFO'
        lam = 0.1 # 0.1
        r = Fp**lam * np.log( (Fp/Fn)**(1-lam)) #  (0.1  89.484%  91.392%)           # sent05 76.9    78.3  
                                              #  (0.05 89.184%  91.536%  91.724%)  # sent05 78.01%  78.9
                                              #  0.04                                       78.05
                                              #  (0.01 88.66%   91.516%  91.892%)  # sent05 78.0    79.0
                                              #   0.007                            # sent05 77.9%   79.2%
    return dic, r  


 
def main(ptrain, ntrain, ptest, ntest, out, liblinear, ngram, sws, w2v):
    ngram = [int(i) for i in ngram]
    print "counting..."
    poscounts = build_dict(ntrain, ngram)         
    negcounts = build_dict(ptrain, ngram)         
    
    dic, r = compute_ratio(poscounts, negcounts, sws=sws)
    print "processing files..."
    process_files(ptrain, ntrain, dic, r, "train-nbsvm.txt", ngram, w2v)
    process_files(ptest, ntest, dic, r, "test-nbsvm.txt", ngram, w2v)
    
    trainsvm = os.path.join(liblinear, "train") 
    predictsvm = os.path.join(liblinear, "predict") 
    # os.system(trainsvm + " train-nbsvm.txt model.logreg")
    # os.system(predictsvm + " test-nbsvm.txt model.logreg " + out)
    os.system(trainsvm + " -s 0 train-nbsvm.txt model.logreg")
    os.system(predictsvm + " -b 1 test-nbsvm.txt model.logreg " + out)
    os.remove("model.logreg")
    os.remove("train-nbsvm.txt")
    os.remove("test-nbsvm.txt")

if __name__ == "__main__":
    """
    Usage :

    python nbsvm.py --liblinear /PATH/liblinear-1.96\
        --ptrain /PATH/data/full-train-pos.txt\
        --ntrain /PATH/data/full-train-neg.txt\
        --ptest /PATH/data/test-pos.txt\
        --ntest /PATH/data/test-neg.txt\
         --ngram 123 --out TEST-SCORE
    """

    parser = argparse.ArgumentParser(description='Run NB-SVM on some text files.')
    parser.add_argument('--liblinear', help='path of liblinear install e.g. */liblinear-1.96')
    parser.add_argument('--ptrain', help='path of the text file TRAIN POSITIVE')
    parser.add_argument('--ntrain', help='path of the text file TRAIN NEGATIVE')
    parser.add_argument('--ptest', help='path of the text file TEST POSITIVE')
    parser.add_argument('--ntest', help='path of the text file TEST NEGATIVE')
    parser.add_argument('--out', help='path and fileename for score output')
    parser.add_argument('--ngram', help='N-grams considered e.g. 123 is uni+bi+tri-grams')
    args = vars(parser.parse_args())

    main(**args)
