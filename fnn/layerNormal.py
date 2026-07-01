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


class LayerNorm(nn.Module):
    def __init__(self,emb_dim,eps=1e-6):
        """
        初始化函数两个参数，一个是emb_dim,表示词嵌入的维度
        一个是eps，表示足够小的数据，防止除零，在规范化公式的分母中数显
        """
        super(LayerNorm,self).__init__()
 

        self.a_2 = nn.Parameter(torch.ones(emb_dim))
        self.b_2 = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        norn = (x-mean)/torch.sqrt((std+self.eps))
        return self.a_2*norn + self.b_2

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
    mah_result = mha(query,key,value,mask)
    # print(mah_result)
    # print(mah_result.shape)

    ffn_layer = FNN(input_dim=embed_dim,hidden_dim=512,output_dim=embed_dim)
    ffn_result = ffn_layer(mah_result)

    eps = 1e-6
    lnorm = LayerNorm(emb_dim=embed_dim)
    lngn_result = lnorm(ffn_result)
    print(lngn_result)
    print(lngn_result.shape)

if __name__ == "__main__":
    main()