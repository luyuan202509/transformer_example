import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import ln
import torch 
import torch.nn as nn 
import torch.functional as F

from data_input.demo1 import Embedding
from data_input.demo2_pos import PositionalEncoding
from encoder.demo3_multi_attention import MultiHeadAttention
from demo1_fnn import FNN
from layerNormal import LayerNorm




class SublayerConnection(nn.Module):
    def __init__(self,emb_dim,dropout=0.01):
        super(SublayerConnection,self).__init__()
        self.layNorm = LayerNorm(emb_dim,eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.layNorm(x)))


def main():
    # 超参
    embed_dim = 512 # 词嵌入维度512维度
    dropout = 0.1
    max_len = 60  #实际
    
    vocab_size = 1000 # 词汇表大小最大1000个词
    
    input = torch.LongTensor([[1,998,4,514],[42,894,2,44],[2,21,600,4]])
    embedding = Embedding(vocab_size,embed_dim)
    emb_res = embedding(input)
    pe = PositionalEncoding(emb_dim=embed_dim,max_len=max_len,dropout=dropout)
    pe_result = pe(emb_res)


    # 
    head = 8 

    # 
    query = key = value = pe_result
    mask = torch.zeros(3,4,4)
    mha = MultiHeadAttention(head,embed_dim,dropout)
    
    sublayer = lambda x:mha(query,key,value,mask)

    sc = SublayerConnection(embed_dim,dropout)
    sc_result = sc(pe_result,sublayer)
    print(sc_result)
    print(sc_result.shape)    


if __name__ == "__main__":
    main()