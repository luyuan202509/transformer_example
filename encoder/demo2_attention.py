import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_input.demo1 import Embedding
from data_input.demo2_pos import PositionalEncoding


def test_maskfill():
    x = torch.randn(5,5)
    print(x)
    mask = torch.zeros(5,5)
    print(mask)
    
    y = x.masked_fill(mask==0,float('-inf'))
    print(y)


def attention(query,key,value,mask:torch.Tensor=None,dropout=None):
    # query 输入的矩阵
    # key 输入的矩阵
    # value 输入的矩阵
    # mask 掩码矩阵
    # dropout c传入的dropout实例化对象
    d_k = query.size(-1)
    
    score = torch.matmul(query,key.transpose(-2,-1)) / np.sqrt(d_k)
    # score = (query @ key.T) / np.sqrt(d_k)

    if mask is not None:
        #score = score.masked_fill(mask==0,float('-inf'))
        score = score.masked_fill(mask==0,-1e9)

    p_attention= F.softmax(score,dim=-1)
    
    if dropout is not None:
        p_attention = dropout(p_attention)
    
    return torch.matmul(p_attention,value),p_attention


def test_attention():
    emb_dim = 512 # 词嵌入维度512维度
    dropout = 0.1
    max_len = 60  #实际
    
    vocab_size = 1000 # 词汇表大小最大1000个词
    
    input = torch.LongTensor([[1,998,4,514],[42,894,2,44],[2,21,600,4]])

    print("词嵌入"+"=="*20)
    embedding = Embedding(vocab_size,emb_dim)
    emb_input = embedding(input)
    print(emb_input)
    print(emb_input.shape)
    print("=="*20)
    print("位置编码" + "=="*20)
    pe = PositionalEncoding(emb_dim,dropout,max_len)
    pe_result = pe(emb_input)
    print(pe_result)
    print(pe_result.shape)

    print("注意力计算"+"=="*20)
    query = key = value = pe_result
    mask = torch.zeros(3,4,4)
    attention_result,attention_result_p = attention(query,key,value,mask=mask)
    print(attention_result)
    print(attention_result.shape)
    print(attention_result_p)
    print(attention_result_p.shape)

    """
# 方案一：不传 mask（最简，双向注意力）
attention_result, attention_result_p = attention(query, key, value)
# 方案二：显式允许所有位置互看
mask = torch.ones(3, 4, 4)
attention_result, attention_result_p = attention(query, key, value, mask=mask)
# 方案三：因果掩码（只能看到自己及左边的 token）
mask = torch.tril(torch.ones(3, 4, 4))
attention_result, attention_result_p = attention(query, key, value, mask=mas

这个掩码矩阵会影响输入的 query 和 key 的矩阵， 

    """
    


    

if __name__ == '__main__':
    test_attention()