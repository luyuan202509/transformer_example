import torch 
from torch.utils.data import Dataset, DataLoader
from torch import nn
from copy import deepcopy
import numpy as np

vocab_list = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n', 
 'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
char_lst = ['a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n', 
 'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
bos_token = "[BOS]"
eos_token = "[EOS]"
pad_token = "[PAD]"

def process(source,target):
    """ 统一长度，转换成数字编码"""
    # 序列长度，最多处理12个字符
    max_length = 12
    # ===================== 长度超过，长度截取 ================================
    if len(source) > max_length:
        source = source[:max_length]
    if len(target)> max_length -1:
        target = target[:max_length -1]
    # ===================== 转换成数字编码，并添加[BOS]和[EOS] ================================
    source_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index(p) for p in target]
    target_id = [vocab_list.index(bos_token)] + target_id + [vocab_list.index(eos_token)]


   # =====================长度不够，补全 ================================
    source_mask = np.array([1] * max_length)
    target_mask = np.array([1] * (max_length +1))
    # source 补全长度并掩码
    if len(target_id) < max_length:
        pad_len = max_length - len(source_id) # 补全的长度
        source_id = source_id + [vocab_list.index(pad_token)] * pad_len
        source_mask[-pad_len:] = 0 # 补全的位置mask为0

    # target 补全补全长度并掩码
    if len(target_id) < max_length + 1:
        pad_len = max_length - len(target_id) + 1 # 补全的长度
        target_id = target_id + [vocab_list.index(pad_token)] * pad_len
        target_mask[-pad_len:] = 0 # 补全的位置mask为0
    
    return source_id,source_mask,target_id,target_mask

class MyDataset(Dataset):
    def __init__(self, source_path, target_path)->None:
        super().__init__()
        self.source_list = []
        self.target_list = []
        with open(source_path, 'r') as f:
            for line in f:
                self.source_list.append(deepcopy(line.strip()))
        with open(target_path, 'r') as f:
            for line in f:
                self.target_list.append(deepcopy(line.strip()))

    def __len__(self):
        return len(self.source_list)
    def __getitem__(self, index):
        source_id,source_mask,target_id,target_mask = process(self.source_list[index],self.target_list[index])
        return (torch.tensor(source_id,dtype=torch.long),torch.tensor(source_mask,dtype=torch.bool),
                torch.tensor(target_id,dtype=torch.long),torch.tensor(target_mask,dtype=torch.bool))

if __name__ == "__main__":
    test_date = MyDataset("source.txt", "target.txt")
    source_id,source_mask,target_id,target_mask = test_date[2]
    