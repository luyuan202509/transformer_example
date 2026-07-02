from this import d
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from common import Encoder,Decoder,Genarator,Embedding,FNN
from common import EncoderLayer,DecoderLayer,MultiHeadAttention




class EncoderDecoder(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,source_embed:Embedding,
                      target_embed:Embedding,generator:Genarator):

        super(EncoderDecoder,self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.source_embed = source_embed
        self.target_embd = target_embed

    def forward(self,source:torch.Tensor,target:torch.Tensor,src_mask:torch.Tensor,target_mask:torch.Tensor):
        
        return self.decode(self.encode(source,src_mask),src_mask,target,target_mask)

    def encode(self,source:torch.Tensor,src_mask:torch.Tensor):
        return self.encoder(self.source_embed(source),src_mask)
        
    def decode(self,memory:torch.Tensor,src_mask:torch.Tensor,target:torch.Tensor,target_mask:torch.Tensor):
        
        return self.decoder(self.target_embd(target),memory,src_mask,target_mask)


def main():

    # 实例化参数 
    vocab_size = 1000
    emb_dim = 512
    dropout_rate = 0.1
    
    source_embed = Embedding(vocab_size,emb_dim)
    target_embed = Embedding(vocab_size,emb_dim)
    generator = Genarator(emb_dim,vocab_size)

    multi_att = MultiHeadAttention(4,emb_dim)
    ffn = FNN(emb_dim,emb_dim,emb_dim)
    encoder_layer = EncoderLayer(emb_dim,multi_att,ffn,dropout_rate)
    decoder_layer = DecoderLayer(emb_dim,multi_att,multi_att,ffn,dropout_rate)
    encoder = Encoder(encoder_layer,3)
    decoder = Decoder(decoder_layer,3)
 

    # 输出参数 
    source = target =torch.LongTensor([[1,998,4,514],[42,894,2,44],[2,21,600,4]])
    source_mask = target_mask = torch.zeros(3,4,4)
    
    ed = EncoderDecoder(encoder,decoder,source_embed,target_embed,generator)
    ed_result = ed(source,target,source_mask,target_mask)
    print(ed_result)
    print(ed_result.shape)

    

if __name__ == "__main__":
    main()
    

