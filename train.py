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
    print(f"input_id: {input_id.shape}, input_m: {input_m.shape}, output_id: {output_id.shape}, output_m: {output_m.shape}")
    output = model(input_id,input_m,output_id[:, :-1],output_m[:, :-1]) 
    print(f"output: {output.shape}")
    break