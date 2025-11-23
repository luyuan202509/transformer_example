from torch.utils.data import DataLoader
from data import MyDataset
from model import Transformer
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import Adam




vocab_list = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n', 
 'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']

from data import process as process_data

def predict(model:Transformer,sentence:str,device:str):
    max_len = 12
    source_id, source_m, _, _  = process_data(sentence, "aaaaa")
    target_id_lst = [vocab_list.index("[BOS]")]
    target_m_lst = [1]
    source_id = torch.tensor(source_id).to(device).unsqueeze(0)
    source_m = torch.tensor(source_m).to(device).unsqueeze(0)
    for _ in range(max_len):
        target_id = torch.tensor(target_id_lst).to(device).unsqueeze(0)
        target_m = torch.tensor(target_m_lst).to(device).unsqueeze(0)
        output = model(source_id, source_m, target_id, target_m)
        word_id = torch.argmax(output[0][-1])
        target_id_lst.append(word_id.item())
        target_m_lst.append(1)
        if word_id == vocab_list.index("[EOS]"):
            break
    result = ""
    for id in target_id_lst:
        result += vocab_list[id]
    return result
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    my_model = Transformer()
    my_model.load_state_dict(torch.load("model.pth"))
    my_model.to(device)
    sentence = "abcdif"
    print("翻译结果: ", predict(my_model,sentence,device))
