import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import MyDataset
from model import Transformer

model = Transformer()
dataset = MyDataset("source.txt", "target.txt")
train_loader = DataLoader(dataset,batch_size=2,shuffle=True)
for input_id,input_m,output_id,output_m in train_loader:
    model(input_id,input_m,output_id[:, :-1],output_m[:, :-1]) 
    pass 