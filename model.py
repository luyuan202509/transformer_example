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
    pass      

class Transformer_block(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Transformer_block, self).__init__(*args,**kwargs)
        self.W_q = nn.Linear(24, 24,bias=False)
        self.W_k = nn.Linear(24, 24,bias=False)
        self.W_v = nn.Linear(24, 24,bias=False)
        self.W_o = nn.Linear(24, 24,bias=False)
    def forward(self,x:torch.Tensor):
        Q,K,V = self.W_q(x),self.W_k(x),self.W_v(x)
        O = attention(Q,K,V)
        O = self.W_o(O)
        return O


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
    