import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class Genarator(nn.Module):
    def __init__(self,emb_dim:int,vocab_size:int):
        super(Genarator,self).__init__()
        self.project = nn.Linear(emb_dim,vocab_size)

    def forward(self,x):
        return F.log_softmax(self.project(x),dim=-1)


def main():
    emb_dim = 512
    vocab_size = 1000
    gen = Genarator(emb_dim,vocab_size) 
    
    decoder_result = torch.randn(3,4,512)
    
    gen_result = gen(decoder_result)
    print(gen_result)
    print(gen_result.shape) 
    
if __name__ == "__main__":
    main()