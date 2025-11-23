import torch 
from torch import nn


class EBD(nn.Module):
    def __init__(self,*args,**kwargs)->None:
        super(EBD, self).__init__(*args,**kwargs)
        self.embedding = nn.Embedding(28, 24)
        self.pos_embedding = nn.Embedding(12, 24)
        self.pos_t = torch.arange(0, 12).reshape(1,12)

    def forward(self, x:torch.Tensor):
        return self.embedding(x) + self.pos_embedding(self.pos_t[:,:x.shape[-1]])
def attention(Q,K,V,M:torch.Tensor):
    A = Q @ K.transpose(-1,-2) / (K.shape[-1] ** 0.5)
    M = M.unsqueeze(1)
    A = A.masked_fill(M == 0,float('-inf'))
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

class Attention_block(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Attention_block, self).__init__(*args,**kwargs)
        self.W_q = nn.Linear(24, 24,bias=False)
        self.W_k = nn.Linear(24, 24,bias=False)
        self.W_v = nn.Linear(24, 24,bias=False)
        self.W_o = nn.Linear(24, 24,bias=False)
    def forward(self,x:torch.Tensor,M:torch.Tensor):
        Q,K,V = self.W_q(x),self.W_k(x),self.W_v(x)
        Q,K,V = transpose_qkv(Q),transpose_qkv(K),transpose_qkv(V)
        O = attention(Q,K,V,M)
        O = transpose_o(O)
        O = self.W_o(O)
        return O

# 归一化层
class Add_Norm(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Add_Norm, self).__init__(*args,**kwargs)
        self.layernorm = nn.LayerNorm(24)
        self.dropout = nn.Dropout(0.1)
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
        x = self.dropout(x)
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
        self.attention = Attention_block()
        self.add_norm_1 = Add_Norm()
        self.pos_ffn = Pos_FFN()
        self.add_norm_2 = Add_Norm()
    def forward(self,x:torch.Tensor,I_m:torch.Tensor):
        I_m = I_m.unsqueeze(-2) # 添加一个维度，用于掩码
        x1 = self.attention(x,I_m)
        x1 = self.add_norm_1(x,x1)
        x2 = self.pos_ffn(x1)
        x2 = self.add_norm_2(x1,x2)
        return x2

# 编码器组装
class Encoder(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Encoder, self).__init__(*args,**kwargs)
        self.ebd = EBD()
        self.encoder_blocks = nn.Sequential()
        self.encoder_blocks.append(Encoder_block())
        self.encoder_blocks.append(Encoder_block())
    def forward(self,x:torch.Tensor,I_m:torch.Tensor):
        ebd_x = self.ebd(x)
        for encoder_block in self.encoder_blocks:
            output = encoder_block(ebd_x,I_m)
        return output


class CrossAttention_block(nn.Module):
    """交叉注意力机制"""
    def __init__(self, *args,**kwargs)->None:
        super(CrossAttention_block, self).__init__(*args,**kwargs)
        self.W_q = nn.Linear(24, 24,bias=False)
        self.W_k = nn.Linear(24, 24,bias=False)
        self.W_v = nn.Linear(24, 24,bias=False)
        self.W_o = nn.Linear(24, 24,bias=False)

    def forward(self,x,x_en):
        Q,K,V = self.W_q(x),self.W_k(x_en),self.W_v(x_en)
        Q,K,V = transpose_qkv(Q),transpose_qkv(K),transpose_qkv(V)
        O = attention(Q,K,V)
        O = transpose_o(O)
        O = self.W_o(O)
        return O

# 解码器块
class Decoder_block(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Decoder_block, self).__init__(*args,**kwargs)
        self.attention = Attention_block()
        self.add_norm_1 = Add_Norm()
        self.cross_attention = CrossAttention_block()
        self.add_norm_2 = Add_Norm()
        self.pos_ffn = Pos_FFN()
        self.add_norm_3 = Add_Norm()

        mask = torch.tril(torch.ones(12, 12))
        self.trail_mask = torch.tril(mask).unsqueeze(0)
    def forward(self,X_t,O_m,X_en,I_m):
        O_m = O_m.unsqueeze(-2)
        X_1 = self.attention(X_t,O_m * self.trail_mask)
        X_t= self.add_norm_1(X_t,X_1)
        X_1  = self.cross_attention(X_t,X_en)
        X_t = self.add_norm_2(X_t,X_1)
        X_1 = self.pos_ffn(X_t)
        X_t = self.add_norm_3(X_t,X_1)
        return X_t

# 解码器
class Decoder(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Decoder, self).__init__(*args,**kwargs)
        self.ebd = EBD()
        self.decoder_blocks = nn.Sequential()
        self.decoder_blocks.append(Decoder_block())
        self.decoder_blocks.append(Decoder_block())
        self.linear = nn.Linear(24, 28,bias=False)

    def forward(self,X_t,O_m,X_en,I_m):
        X_t = self.ebd(X_t)
        for decoder_block in self.decoder_blocks:
            X_t = decoder_block(X_t,O_m,X_en,I_m)
        X_t = self.linear(X_t)
        return X_t

# 构建模型
class Transformer(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Transformer, self).__init__(*args,**kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self,X_s,I_m,X_t,O_m):
        x_en = self.encoder(X_s,I_m)
        x = self.decoder(X_t,O_m,x_en,I_m)
        return x
        

def main1():
    print("\n==词嵌入====================================================================\n")
    aaa = torch.ones((2,12)).long()
    print(f"aaa.shape: {aaa.shape}")
    embedding = EBD()
    aaa = embedding(aaa)
    print(f"aaa.shape: {aaa.shape}")
    
    print("\n==注意力机制====================================================================\n")
    atten_en = Attention_block()
    aaa = atten_en(aaa)
    print(f"计算注意力后的aaa.shape: {aaa.shape}")

def main2():
    print("\n==编码器====================================================================\n")
    aaa = torch.ones((2,12)).long()
    print(f"输入aaa.shape: {aaa.shape}")
    encoder = Encoder()
    aaa = encoder(aaa)
    print(f"编码器输出aaa.shape: {aaa.shape}")
    print(f"编码器输出aaa: {aaa}")
    print(f"encoder: {encoder}")

def main3():
    """测试模型"""
    aaa = torch.ones((2,12)).long()
    bbb = torch.ones((2,4)).long()
    print(f"输入aaa.shape: {aaa.shape}")
    model = Transformer()
    output = model(aaa,bbb)
    print(f"输出output.shape: {output.shape}")
   

if __name__ == "__main__":  
    #main1()
    #main2()
    main3()