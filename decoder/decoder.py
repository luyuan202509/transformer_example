import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import torch.nn as nn
from data_input.demo1 import Embedding
from data_input.demo2_pos import PositionalEncoding
from encoder.demo3_multi_attention import MultiHeadAttention
from fnn.demo1_fnn import FNN
from encoder.encoder import Encoder,EncoderLayer
import numpy as np

def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


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

class SublayerConnection(nn.Module):
    def __init__(self,emb_dim,dropout=0.01):
        super(SublayerConnection,self).__init__()
        self.layNorm = LayerNorm(emb_dim,eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x,sublayer):
        # x: 输入张量
        # sublayer: 一个函数，函数中执行网络层的计算
        sub = sublayer(self.layNorm(x)) # 子层网络计算，这里使用前置层归一化
        return x + self.dropout(sub)

        #return x + self.dropout(sublayer(self.layNorm(x)))

class DecoderLayer(nn.Module):
    def __init__(self,embed_dim,self_attn,src_attn,ffn,dropout):
        super(DecoderLayer,self).__init__()
        # embed_dim: 词嵌入维度
        # self_attn: 自注意力对象
        # src_attn: 常规注意力机制对象
        # ffn ： 前馈神经网络对象
        # dropout: 丢弃概率
        self.embed_dim = embed_dim
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ffn = ffn
        self.dropout = dropout

        self.sublayer = clones(SublayerConnection(embed_dim,dropout),3)

    def forward(self,x,memory,src_mask,target_mask):
        # x: 目标序列的词嵌入
        # memory: 编码器的语义存储张量
        # src_mask: 编码器输入序列的填充掩码张量
        # target_mask: 目标序列的填充掩码张量
        m = memory
        # x 先过多头自注意力层 target_mask 掩码让模型不能查看未来的信息
        att_fun = lambda x: self.self_attn(x,x,x,target_mask)
        x = self.sublayer[0](x,att_fun)

        # x 经过交叉多头注意力层，q!=k=v; k,v来自编码器层的输出 src_mask 遮掩对结果无用的信息
        att_fun = lambda x: self.src_attn(x,m,m,src_mask)
        x = self.sublayer[1](x,att_fun)
        
        # 经过前馈神经网络层    
        output = self.sublayer[2](x,self.ffn)
    
        return output
        

class Decoder(nn.Module):
    def __init__(self,coder_layer:DecoderLayer,num_layer:int):
        super(Decoder,self).__init__()
        self.layers = clones(coder_layer,num_layer)
        self.num_layer = num_layer
        self.norm = LayerNorm(coder_layer.embed_dim)
    
    def forward(self,x,memory,src_mask,target_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,target_mask)
        return self.norm(x)

def main():
    
    # 实例化参数
    vocab_size = 1000 # 词表大小
    embed_dim = 512
    num_head = 8
    hidden_dim = 64
    dropout = 0.1
    eps = 1e-6
    
    self_attn = src_attn = MultiHeadAttention(num_head,embed_dim,dropout)
    ffn = FNN(embed_dim,hidden_dim,embed_dim)
    
    input = torch.LongTensor([[1,998,4,514],[42,894,2,44],[2,21,600,4]])
    embedding = Embedding(vocab_size,embed_dim)
    emb = embedding(input)
    pos_encoding = PositionalEncoding(embed_dim,dropout)
    pos_emb = pos_encoding(emb)

    mask = torch.zeros(3,4,4)
    src_mask = target_mask = mask 

    encoderLayer = EncoderLayer(embed_dim,self_attn,ffn,dropout)
    encoder = Encoder(encoderLayer,8)
   # 编码器输出
    encoder_result = encoder(pos_emb,src_mask)

    # 解码器层
    dp = copy.deepcopy 
    decoderLayer = DecoderLayer(embed_dim,dp(self_attn),dp(src_attn),dp(ffn),dropout)
    decoder = Decoder(decoderLayer,8)
   
    
    # decoder 输出结果
    decoder_result = decoder(pos_emb,encoder_result,src_mask,target_mask)
    print(decoder_result)
    print(decoder_result.shape)


    

    
    

    
    

    

    


if __name__ == "__main__":
    main()
    