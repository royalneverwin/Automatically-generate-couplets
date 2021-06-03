import torch
import random
import numpy as np
import torch.nn as nn

def collate_fn(batch):
    lengths = [f[2] for f in batch] #1个batch里每个对联的实际长度
    max_len=max(lengths)
    ranks=list(np.argsort(lengths))
    ranks.reverse()
    lengths.sort(reverse=True)
    input_ids = [batch[i][0]+[0]*(max_len-len(batch[i][0])) for i in ranks]  # 将batch里的对联上联都补齐到最大对联对应的长度
    mask = [[1.0]*len(batch[i][0])+[0.0]*(max_len-len(batch[i][0])) for i in ranks] # 对应batch里每一句对联的bool索引
    output_ids = [batch[i][1]+[0]*(max_len-len(batch[i][1])) for i in ranks] # batch下联
    output_ids_flatten = []
    for i in ranks:
        output_ids_flatten.extend(batch[i][1])
    return input_ids, output_ids, lengths, mask, output_ids_flatten

