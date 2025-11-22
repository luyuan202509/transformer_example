import torch 
from torch import nn


class EBD(nn.Module):
    def __init__(self,*args,**kwargs)->None:
        super(EBD, self).__init__(*args,**kwargs)
        self.embedding = nn.Embedding(28, 24)
        self.pos_embedding = nn.Embedding(12, 24)
        self.pos_t = torch.arange(0, 12).reshape(1,12)

    def forward(self, x:torch.Tensor):
        return self.embedding(x) + self.pos_embedding(self.pos_t)
def attention(Q,K,V):
    A = Q @ K.transpose(-1,-2) / (K.shape[-1] ** 0.5)
    A = torch.softmax(A,dim=-1)
    O = A @ V
    return O

def transpose_o(O:torch.Tensor):
    """处理注意力输出 W_o 的维度"""
    O=O.transpose(-2,-3)
    O=O.reshape(O.shape[0],O.shape[1],-1) # 前两个维度不变，合并最后两个维度
    return O
def transpose_qkv(qkv:torch.Tensor):
    qkv= qkv.reshape(qkv.shape[0],qkv.shape[1],4,6) # 嵌入维度24 拆成4,6
    qkv = qkv.transpose(-2,-3) # 交换最后两个维度
    return qkv

class Transformer_block(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Transformer_block, self).__init__(*args,**kwargs)
        self.W_q = nn.Linear(24, 24,bias=False)
        self.W_k = nn.Linear(24, 24,bias=False)
        self.W_v = nn.Linear(24, 24,bias=False)
        self.W_o = nn.Linear(24, 24,bias=False)

    def forward(self,x:torch.Tensor):
        Q,K,V = self.W_q(x),self.W_k(x),self.W_v(x)
        Q,K,V = transpose_qkv(Q),transpose_qkv(K),transpose_qkv(V)
        O = attention(Q,K,V)
        O = transpose_o(O)
        O = self.W_o(O)
        return O


# 归一化层
class Add_Norm(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Add_Norm, self).__init__(*args,**kwargs)
        self.layernorm = nn.LayerNorm(24)
    def forward(self,x,x1:torch.Tensor):
        """
        参数：
        x: 嵌入向量 
        x1: 注意力输入
        返回：
        x: 归一化
        """
        x = x + x1
        x = self.layernorm(x)
        return x 

# 位置前馈网络
class Pos_FFN(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Pos_FFN, self).__init__(*args,**kwargs)
        self.linear1  = nn.Linear(24, 48,bias=False)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(48, 24,bias=False)
        self.relu2 = nn.ReLU()

    def forward(self,x:torch.Tensor):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x

# 编码器块
class Encoder_block(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Encoder_block, self).__init__(*args,**kwargs)
        self.attention = Transformer_block()
        self.add_norm_1 = Add_Norm()
        self.pos_ffn = Pos_FFN()
        self.add_norm_2 = Add_Norm()
    def forward(self,x:torch.Tensor):
        x1 = self.attention(x)
        x1 = self.add_norm_1(x,x1)
        x2 = self.pos_ffn(x1)
        x2 = self.add_norm_2(x1,x2)
        return x2

# 编码器组装
class Encoder(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Encoder, self).__init__(*args,**kwargs)
        self.ebd = EBD()




        self.add_norm = Add_Norm()
        self.pos_ffn = Pos_FFN()

        
    def forward(self,x:torch.Tensor):
        x = self.add_norm(x,x)
        x = self.pos_ffn(x)
        return x

if __name__ == "__main__":  
    print("\n==词嵌入====================================================================\n")
    aaa = torch.ones((2,12)).long()
    print(f"aaa.shape: {aaa.shape}")
    embedding = EBD()
    aaa = embedding(aaa)
    print(f"aaa.shape: {aaa.shape}")
    
    print("\n==注意力机制====================================================================\n")
    atten_en = Transformer_block()
    aaa = atten_en(aaa)
    print(f"计算注意力后的aaa.shape: {aaa.shape}")