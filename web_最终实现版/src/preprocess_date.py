import os
from torch.utils.data import Dataset
import torch.nn as nn
import codecs
import numpy as np

"""def build_vocab():
    word2id={'PAD': 0, '“': 1, '”': 2}
    word_cnt=3
    id2word=['PAD', '“', '”']
    with open("./couplets/all.txt","r",encoding='utf-8') as fr:
        raw=fr.readlines()
    for item in raw:
        if item[0]=='\n':
            continue
        item= ['“'] + item.strip().split(' ') + ['”']
        for word in item :
            if word not in word2id:
                word2id[word]=word_cnt
                word_cnt+=1
                id2word.append(word)
    word2id['UNK']=word_cnt
    word_cnt+=1
    id2word.append('UNK')
    return word2id,id2word,word_cnt"""
def load_dense_drop_repeat(path):
    vocab_size, size = 0, 0
    vocab = {}
    vocab["i2w"], vocab["w2i"] = [], {}
    count = 0
    with codecs.open(path, "r", "utf-8") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.rstrip().split()[1])
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            if not vocab["w2i"].__contains__(vec[0]):
                vocab["w2i"][vec[0]] = count
                matrix[count, :] = np.array([float(x) for x in vec[1:]])
                count += 1
    for w, i in vocab["w2i"].items():
        vocab["i2w"].append(w)
    return matrix, vocab, size, len(vocab["i2w"])





