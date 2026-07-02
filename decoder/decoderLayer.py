import copy 
import torch 
import torch.nn as nn

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
        


        
        
        
        