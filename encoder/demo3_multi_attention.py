import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn
import numpy as np
from torch.nn.modules import dropout
from encoder.demo2_attention import attention
import copy

from data_input.demo1 import Embedding
from data_input.demo2_pos import PositionalEncoding

# torch.view演示
def test_view():
   # x = torch.arange(4*5).view(4,5)
    x = torch.randn((4,4))
    d = x.size()[0] * x.size()[1] # 4 * 5 = 20
    y = x.view(d)
    print(y)
    
    z = x.view(-1,8) # 第二个维度写死为8维，第一维自适应 
    print(z)

def test_transpose():
    a = torch.randn(1,2,3,4)
    print(a.size())
    print(a)
    
    print('======================')
    b = a.transpose(1,2)
    print(b.size())
    print(b)
    
    print("======================")
    c = a.view(1,3,2,4)
    print(c.size())
    print(c)



def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadAttention(nn.Module):
    def __init__(self,head,embed_dim,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        assert embed_dim % head == 0
        self.head = head
        self.embed_dim = embed_dim
        self.d_k = embed_dim // head

        # 获得线性层 q,k,v,和输出层的权重矩阵
        self.linears = clones(nn.Linear(embed_dim,embed_dim),4)
        self.attn = None 
        self.dropout = nn.Dropout(p=dropout)
        
            
    def forward(self,query,key,value,mask=None):
        # query,key,value的维度为 (batch_size,seq_len,embed_dim)

        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        
        query,value,key = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
            for model,x in zip(self.linears,(key,value,query))]
        
        # 将每个头输出的输出传入到注意力层
        x,self.attn = attention(query,key,value,mask=mask,dropout = self.dropout)
        
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)

        return self.linears[-1](x)  

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
    embed_dim = 512
    dropout = 0.1

    # 
    query = key = value = pe_result
    mask = torch.zeros(3,4,4)
    mha = MultiHeadAttention(head,embed_dim,dropout)
    mah_result = mha(query,key,value,mask)
    print(mah_result)
    print(mah_result.shape)

if __name__ == '__main__':
    main()     