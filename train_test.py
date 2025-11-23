from torch.utils.data import DataLoader
from data import MyDataset
from model import Transformer
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import Adam


device  = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
my_model = Transformer().to(device)
dataset = MyDataset("source.txt", "target.txt")
dataloader = DataLoader(dataset,batch_size=32,shuffle=True)
loss_func = nn.CrossEntropyLoss(ignore_index=2)
trainer = Adam(params=my_model.parameters(), lr=0.0005)

for epoch in range(200):
    t = tqdm(dataloader)
    for input_id,input_m,output_id,output_m in t:
        output = my_model(input_id.to(device), input_m.to(device), output_id[:, :-1].to(device), output_m[:, :-1].to(device))
        target = output_id[:, 1:].to(device)
        loss = loss_func(output.reshape(-1, 29), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1)
        trainer.step()
        trainer.zero_grad()
        # print(loss.item())
        t.set_description(str(loss.item()))

torch.save(my_model.state_dict(), "model.pth")
        