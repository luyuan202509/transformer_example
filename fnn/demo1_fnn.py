import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn 
import torch.functional as F

from data_input.demo1 import Embedding
from data_input.demo2_pos import PositionalEncoding
from encoder.demo3_multi_attention import MultiHeadAttention

class FNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(FNN,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


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
    print(ffn_result)
    print(ffn_result.shape)    


if __name__ == "__main__":
    main()