import numpy as np  
import torch
import torch.nn as nn 

def subsequence_mask(size):
    attn_shape = (1,size, size)
    attn_mask = np.triu (np.ones(attn_shape),k=1).astype('uint8')
    #attn_mask = np.triu(np.ones(attn_shape),k=0).astype('uint8')
    #attn_mask = np.triu(np.ones(attn_shape),k=-1).astype('uint8')
    
    turn_mask = torch.from_numpy(1-attn_mask)
    print(turn_mask)





if __name__ == "__main__":
    subsequence_mask(5) 