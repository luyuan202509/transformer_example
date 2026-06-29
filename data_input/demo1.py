#从 PyTorch 0.4.0（2018年4月）起，Variable 和 Tensor 已经合并。 现在直接使用 torch.Tensor 即可，它本身就具备 Variable 的全部功能
from torch.autograd import Variable  
import torch
import torch.nn as nn
import math
# embedding = nn.Embedding(num_embeddings=10,embedding_dim=3)
# input = torch.LongTensor([[1,2,4,5],[4,3,2,9],[2,3,6,4],[9,3,5,6],[1,2,3,4]])
# emb = embedding(input)
# print(emb)
# print(emb.shape)

# # 始终为零向量 [0, 0, 0]，不参与学习
# embedding = nn.Embedding(num_embeddings=10,embedding_dim=3,padding_idx=0)
# input = torch.LongTensor([[0,0,4,5],[4,3,0,0]])
# emb = embedding(input)
# print(emb) 
# print(emb.shape)


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

def main():
    vocab_size = 1000
    emb_dim = 512
    
    input = torch.LongTensor([[1,998,4,514],[42,894,2,44],[2,21,600,4]])
    
    embedding = Embedding(vocab_size,emb_dim)
    emb = embedding(input)
    print(emb)
    print(emb.shape)

if __name__ == "__main__":
    main()
        