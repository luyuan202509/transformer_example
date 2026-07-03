"""
Multi30k 英→德翻译数据集加载器。

数据来源：Multi30k (Flickr30k) — WMT 2016 Multimodal Task
下载地址：https://gitcode.com/open-source-toolkit/66124

数据集统计：
    train:  29,001 平行句对
    val:     1,015 平行句对
    test:    1,000 平行句对
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ── 特殊 token ─────────────────────────────────
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ═══════════════════════════════════════════════════════════════════
# 词汇表
# ═══════════════════════════════════════════════════════════════════

class Vocabulary:
    """词 → ID 映射表。"""

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = list(SPECIAL_TOKENS)   # index → token
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # token → index

    def __len__(self):
        return len(self.itos)

    def build_from_sentences(self, sentences):
        """从句子列表构建词汇表（按频率过滤）。"""
        counter = Counter()
        for sent in sentences:
            counter.update(sent.strip().split())

        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def encode(self, sentence):
        """将句子转为 token ID 列表，添加 <s> 和 </s>。"""
        tokens = [SOS_IDX]
        for word in sentence.strip().split():
            tokens.append(self.stoi.get(word, UNK_IDX))
        tokens.append(EOS_IDX)
        return tokens

    def decode(self, ids, skip_special=True):
        """将 token ID 列表转回句子。"""
        tokens = []
        for i in ids:
            if skip_special and i in (PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX):
                continue
            tokens.append(self.itos[i] if i < len(self.itos) else UNK_TOKEN)
        return ' '.join(tokens)


# ═══════════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """平行语料数据集。

    Parameters
    ----------
    src_path : str — 源语言文件路径（英文）
    tgt_path : str — 目标语言文件路径（德文）
    src_vocab : Vocabulary — 源语言词汇表
    tgt_vocab : Vocabulary — 目标语言词汇表
    """

    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        with open(src_path, 'r', encoding='utf-8') as f:
            self.src_sentences = [line.strip() for line in f]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [line.strip() for line in f]

        assert len(self.src_sentences) == len(self.tgt_sentences), \
            f"源/目标行数不一致: {len(self.src_sentences)} vs {len(self.tgt_sentences)}"

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_ids = self.src_vocab.encode(self.src_sentences[idx])
        tgt_ids = self.tgt_vocab.encode(self.tgt_sentences[idx])
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════════
# 批次整理
# ═══════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """将不等长句子填充到相同长度，组成 batch。

    Returns
    -------
    src : (batch, src_max_len) — 填充后的源序列
    tgt : (batch, tgt_max_len) — 填充后的目标序列
    """
    src_list, tgt_list = zip(*batch)

    # 源语言填充
    src_max_len = max(len(s) for s in src_list)
    src_padded = torch.full((len(src_list), src_max_len), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(src_list):
        src_padded[i, :len(s)] = s

    # 目标语言填充
    tgt_max_len = max(len(t) for t in tgt_list)
    tgt_padded = torch.full((len(tgt_list), tgt_max_len), PAD_IDX, dtype=torch.long)
    for i, t in enumerate(tgt_list):
        tgt_padded[i, :len(t)] = t

    return src_padded, tgt_padded


# ═══════════════════════════════════════════════════════════════════
# 主入口：加载全部数据
# ═══════════════════════════════════════════════════════════════════

def load_multi30k(data_dir, batch_size=32, min_freq=2, num_workers=0):
    """加载 Multi30k 数据集，构建词汇表和 DataLoader。

    Parameters
    ----------
    data_dir : str — 数据集根目录（含 train/val/test2016 子目录）
    batch_size : int — 批次大小
    min_freq : int — 词汇表最小词频
    num_workers : int — DataLoader 工作进程数

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    src_vocab, tgt_vocab : Vocabulary
    """
    # 文件路径
    train_src = os.path.join(data_dir, 'train', 'train.en')
    train_tgt = os.path.join(data_dir, 'train', 'train.de')
    val_src   = os.path.join(data_dir, 'val', 'val.en')
    val_tgt   = os.path.join(data_dir, 'val', 'val.de')
    test_src  = os.path.join(data_dir, 'test2016', 'test.en')
    test_tgt  = os.path.join(data_dir, 'test2016', 'test.de')

    # 验证文件存在
    for p in [train_src, train_tgt, val_src, val_tgt, test_src, test_tgt]:
        assert os.path.exists(p), f"文件不存在: {p}"

    # ── 构建词汇表（仅用训练集） ──────────────
    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)

    with open(train_src, 'r', encoding='utf-8') as f:
        src_vocab.build_from_sentences(f.readlines())
    with open(train_tgt, 'r', encoding='utf-8') as f:
        tgt_vocab.build_from_sentences(f.readlines())

    print(f'源词汇表 (EN): {len(src_vocab):,} tokens  (min_freq={min_freq})')
    print(f'目标词汇表 (DE): {len(tgt_vocab):,} tokens  (min_freq={min_freq})')

    # 创建数据集
    train_ds = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_ds   = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    test_ds  = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab)

    print(f'训练集: {len(train_ds):,} 句对')
    print(f'验证集: {len(val_ds):,} 句对')
    print(f'测试集: {len(test_ds):,} 句对')

    # 创建 DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# ═══════════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 数据集路径（相对于项目根目录）
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dataset', 'multi30k'
    )

    print('加载 Multi30k 数据集...')
    print(f'目录: {data_dir}')
    print()

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        load_multi30k(data_dir, batch_size=4, min_freq=2)

    # ── 打印一个 batch 的信息 ────────────────
    print()
    print('=' * 60)
    print('Batch 示例')
    print('=' * 60)
    src, tgt = next(iter(train_loader))
    print(f'src 形状: {src.shape}  (batch, src_max_len)')
    print(f'tgt 形状: {tgt.shape}  (batch, tgt_max_len)')
    print()

    # 解码第一个样本
    for i in range(min(2, len(src))):
        src_sent = src_vocab.decode(src[i].tolist())
        tgt_sent = tgt_vocab.decode(tgt[i].tolist())
        print(f'[{i}] EN: {src_sent}')
        print(f'[{i}] DE: {tgt_sent}')
        print(f'    src_ids: {src[i].tolist()}')
        print(f'    tgt_ids: {tgt[i].tolist()}')
        print()
