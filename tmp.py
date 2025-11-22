import torch 
from torch import nn

print("\n==词嵌入====================================================================\n")
a= torch.tensor([[1,2,3,4],[2,3,3,4]])
print(a)
ebd = torch.nn.Embedding(num_embeddings=5,embedding_dim=24)

# 词嵌入操作，将a中的每个词转换为24维的向量
bbb = ebd(a)
print(bbb)

# 位置嵌入操作，将a中的每个词转换为24维的向量
pos_ebd = torch.nn.Embedding(num_embeddings=10,embedding_dim=24)
pos_bbb = pos_ebd(a)
print(pos_bbb)



print("\n==注意力机制====================================================================\n")
