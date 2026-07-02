
from base64 import encode
from turtle import forward
import torch 
import torch.nn as nn 
import copy
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F



# 构建Embedding类实现文本嵌入
class Embedding(nn.Module):
    def __init__(self,vocab_size,emb_dim,padding_idx=0)->None:
        # vacab_size: 词表大小
        # emb_dim: 词嵌入维度
        super(Embedding,self).__init__()
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim,padding_idx=padding_idx)
    def forward(self,input):
        return self.emb(input)*math.sqrt(self.emb_dim)
# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,emb_dim,dropout,max_len=5000):
        # emb_dim 词嵌入维度
        # dropout 置零比例
        # max_len 每个句子最大长度
        super(PositionalEncoding,self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len,emb_dim)

        # 初始化绝对位置编码矩阵
        position = torch.arange(0,max_len).unsqueeze(dim=1)
        # 变换矩阵,跳跃式变换
        div_term = torch.exp(torch.arange(0,emb_dim,2)* -(math.log(1000.0)/emb_dim))
    
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(dim=0)
        
        # 注册到模型buffer
        self.register_buffer('pe',pe)
        
        
    def forward(self,x):
        # pe中最长句子长度太长，截取到和句子的长度一致，
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

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


# 构建子层连接类 残差连接
class SublayerConnection(nn.Module):
    def __init__(self,emb_dim,dropout=0.01):
        super(SublayerConnection,self).__init__()
        self.layNorm = LayerNorm(emb_dim,eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.layNorm(x)))


class EncoderLayer(nn.Module):
    def __init__(self,emb_dim,self_attn,feed_forward,dropout) -> None:
        super(EncoderLayer,self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(emb_dim,dropout),2)
        self.emb_dim = emb_dim
    
    def forward(self,x,mask):
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward) 
def main():

    vocab_size = 1000
    max_len = 500 # 最大句子长度
    emb_dim = 512
    ff_hidden_dim = 64 # 前 feed_forward的隐藏层维度
    dropout = 0.1
    num_head = 8
    

    emblayer =  Embedding(vocab_size,emb_dim)
    pe = PositionalEncoding(emb_dim,dropout,max_len)
    
    self_attn = MultiHeadAttention(num_head,emb_dim,dropout)
    feed_forward = FNN(emb_dim,ff_hidden_dim,emb_dim)
    mask = torch.zeros(3,4,4)

    input = torch.LongTensor([[1,998,4,514],[42,894,2,44],[2,21,600,4]])
    el = EncoderLayer(emb_dim,self_attn,feed_forward,dropout)
    emb_result = emblayer(input)
    pe_result = pe(emb_result)
    
    encoder_result = el(pe_result,mask)

    print(encoder_result)
    print(encoder_result.shape)
    


if __name__ == "__main__":
    main()