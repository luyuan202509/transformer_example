import torch 
from torch.autograd import Variable
import torch.nn as nn 
import math 
from demo1 import Embedding
import matplotlib.pyplot as plt
import numpy as np 



## dropout 演示
def dropout_demo():
    dropoutlayer = nn.Dropout(p = 0.2)

    input = torch.randn(4,5)
    print(input)
    input_2 =  dropoutlayer(input)
    print(input_2)
    
# unsqueeze_demo  张量增加维度 演示
def unsqueeze_demo():
    input = torch.randn(3,4)
    print(input)
    print(input.shape)
    print('====')
    input_new = torch.unsqueeze(input,dim=1)
    print(input_new)
    print(input_new.shape)
    print('====')
    input_new2 = torch.unsqueeze(input,dim=0)
    print(input_new2)
    print(input_new.shape)

    
# 构建位置位置编码器类：Pos
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

def main():
    # 超参
    emb_dim = 512 # 词嵌入维度512维度
    dropout = 0.1
    max_len = 60  #实际
    
    vocab_size = 1000 # 词汇表大小最大1000个词
    
    input = torch.LongTensor([[1,998,4,514],[42,894,2,44],[2,21,600,4]])
    print("词嵌入"+"=="*20)
    embedding = Embedding(vocab_size,emb_dim)
    emb_res = embedding(input)
    print(emb_res)
    print(emb_res.shape)
    print("位置编码"+"=="*20)
    pe = PositionalEncoding(emb_dim=emb_dim,max_len=max_len,dropout=dropout)
    pe_res = pe(emb_res)
    print(pe_res)
    print(pe_res.shape)
    
    
def showPotionView():

    plt.figure(figsize=(15,5))
    
    pe  = PositionalEncoding(20,0)
    y = pe(Variable(torch.zeros(1,100,20)))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())

    plt.legend(["dim %d"%p for p in [4,5,6,7]])
    plt.show()
    
    

        
        

        
           

if __name__ == "__main__":
    showPotionView()
