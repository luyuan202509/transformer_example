"""
Transformer 翻译推理脚本。

用法：
    # 单句翻译
    python predict.py --text "A man is running on the beach."

    # 批量翻译
    python predict.py --input sentences.en.txt --output translations.de.txt

    # 交互模式
    python predict.py --interactive

    # 指定模型和 beam search
    python predict.py --exp_dir experiments/EXP-P2-xxx --beam_size 5 --text "Hello world."
"""
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F

# 添加 model / dataset 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset'))

from transformer import Transformer
from load_data import load_multi30k, Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
from load_data import SPECIAL_TOKENS, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


# ═══════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════

def load_model(exp_dir, checkpoint_name='checkpoint_best.pt', device=None):
    """从实验目录加载模型和词表。

    Parameters
    ----------
    exp_dir : str — 实验目录路径（含 config.json 和 checkpoint）
    checkpoint_name : str — checkpoint 文件名
    device : torch.device or None

    Returns
    -------
    model, src_vocab, tgt_vocab, config
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else
                              'cpu')

    # 加载配置
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'未找到配置文件: {config_path}')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 加载 checkpoint
    ckpt_path = os.path.join(exp_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'未找到 checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)

    # 尝试从 checkpoint 中提取词汇表大小
    ckpt_config = ckpt.get('config', {})
    src_vocab_size = ckpt_config.get('src_vocab_size', config.get('src_vocab_size'))
    tgt_vocab_size = ckpt_config.get('tgt_vocab_size', config.get('tgt_vocab_size'))

    if src_vocab_size is None or tgt_vocab_size is None:
        raise ValueError('无法确定词表大小，请检查 config.json 或 checkpoint')

    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_layers=config.get('num_layers', 6),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', 2048),
        dropout=config.get('dropout', 0.1),
        max_len=5000,
        padding_idx=PAD_IDX,
        tie_embed_weights=config.get('tie_embed_weights', False),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 构建词表（从训练数据重新生成，确保和训练时一致）
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'multi30k')
    _, _, _, src_vocab, tgt_vocab = load_multi30k(
        data_dir, batch_size=32, min_freq=2
    )

    info = {
        'epoch': ckpt.get('epoch', '?'),
        'val_loss': ckpt.get('val_loss', '?'),
        'val_bleu': ckpt.get('val_bleu', '?'),
        'n_params': sum(p.numel() for p in model.parameters()),
        'device': str(device),
    }

    print(f'模型已加载: {os.path.basename(exp_dir)}/{checkpoint_name}')
    print(f'  epoch={info["epoch"]}, val_loss={info["val_loss"]}, '
          f'val_bleu={info["val_bleu"]}')
    print(f'  参数: {info["n_params"]:,}  |  设备: {info["device"]}')
    print(f'  源词表: {len(src_vocab):,}  |  目标词表: {len(tgt_vocab):,}')
    print()

    return model, src_vocab, tgt_vocab, config


# ═══════════════════════════════════════════════════════════════════
# 解码策略
# ═══════════════════════════════════════════════════════════════════

def create_masks(src, tgt, pad_idx=PAD_IDX):
    """构造注意力掩码（和 train.py 中一致）。"""
    device = src.device
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=device), diagonal=1
    ) == 0
    tgt_mask = tgt_pad_mask & causal_mask
    return src_mask, tgt_mask


@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    """贪心解码：每步取概率最高的 token。"""
    model.eval()
    memory = model.encode(src, src_mask)
    batch_size = src.size(0)
    device_local = src.device

    ys = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device_local)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device_local)

    for _ in range(max_len - 1):
        _, tgt_mask = create_masks(src, ys)
        decoded = model.decode(ys, memory, src_mask, tgt_mask)
        log_probs = model.generator(decoded[:, -1:, :])
        next_token = log_probs.argmax(dim=-1)  # (batch, 1)

        ys = torch.cat([ys, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == end_symbol)

        if finished.all():
            break

    return ys


@torch.no_grad()
def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol,
                       beam_size=5, length_penalty=0.6):
    """Beam Search 解码：维护 beam_size 个最佳候选序列。

    使用长度惩罚避免偏向短序列：
        score = sum(log_probs) / (len^alpha)

    Parameters
    ----------
    beam_size : int — 束宽
    length_penalty : float — 长度惩罚指数（0 = 无惩罚，1 = 全惩罚）
    """
    model.eval()
    device_local = src.device
    batch_size = src.size(0)

    # 编码只需一次
    memory = model.encode(src, src_mask)

    # 每个 batch 样本独立做 beam search
    results = []
    for b in range(batch_size):
        # 单个样本的 memory 和 mask
        mem_b = memory[b:b+1]                 # (1, src_len, d_model)
        src_mask_b = src_mask[b:b+1]          # (1, 1, 1, src_len)

        # beam: list of (sequence, log_prob_sum, finished)
        # 初始 beam
        beams = [([start_symbol], 0.0, False)]

        for _ in range(max_len - 1):
            all_candidates = []

            for seq, score, finished in beams:
                if finished:
                    # 已完成的序列直接保留
                    all_candidates.append((seq, score, finished))
                    continue

                # 构造输入
                ys = torch.tensor([seq], dtype=torch.long, device=device_local)
                _, tgt_mask = create_masks(src_mask_b.expand(1, -1, -1, -1), ys)

                decoded = model.decode(ys, mem_b, src_mask_b, tgt_mask)
                log_probs = model.generator(decoded[:, -1, :])  # (1, vocab)
                log_probs = log_probs.squeeze(0)                # (vocab,)

                # 取 top-k
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    token = top_indices[i].item()
                    lp = top_log_probs[i].item()
                    new_seq = seq + [token]
                    new_score = score + lp
                    new_finished = finished or (token == end_symbol)
                    all_candidates.append((new_seq, new_score, new_finished))

            # 按分数排序，保留 top beam_size
            all_candidates.sort(
                key=lambda x: _beam_score(x[1], len(x[0]) - 1, length_penalty),
                reverse=True
            )
            beams = all_candidates[:beam_size]

            # 所有 beam 都结束了
            if all(f for _, _, f in beams):
                break

        # 选择最佳序列（带长度惩罚）
        best_seq, best_score, _ = max(
            beams,
            key=lambda x: _beam_score(x[1], len(x[0]) - 1, length_penalty)
        )
        results.append(best_seq)

    # 填充到相同长度
    max_out_len = max(len(seq) for seq in results)
    padded = torch.full((batch_size, max_out_len), PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(results):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return padded


def _beam_score(log_prob_sum, length, length_penalty):
    """Beam search 评分：带长度惩罚的对数概率均值。"""
    if length == 0:
        return log_prob_sum
    return log_prob_sum / (length ** length_penalty)


# ═══════════════════════════════════════════════════════════════════
# 翻译接口
# ═══════════════════════════════════════════════════════════════════

class Translator:
    """翻译器：封装模型 + 词表 + 解码策略。"""

    def __init__(self, model, src_vocab, tgt_vocab, device=None,
                 beam_size=1, max_len=100):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device or next(model.parameters()).device
        self.beam_size = beam_size
        self.max_len = max_len

    def translate(self, sentence):
        """翻译单句。

        Parameters
        ----------
        sentence : str — 英文句子

        Returns
        -------
        str — 德文翻译
        """
        # 编码
        src_ids = self.src_vocab.encode(sentence)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_mask = (src_tensor != PAD_IDX).unsqueeze(1).unsqueeze(2)

        # 解码
        if self.beam_size <= 1:
            output = greedy_decode(
                self.model, src_tensor, src_mask,
                self.max_len, SOS_IDX, EOS_IDX
            )
        else:
            output = beam_search_decode(
                self.model, src_tensor, src_mask,
                self.max_len, SOS_IDX, EOS_IDX,
                beam_size=self.beam_size
            )

        return self.tgt_vocab.decode(output[0].tolist())

    def translate_batch(self, sentences, show_progress=True):
        """批量翻译。

        Parameters
        ----------
        sentences : list[str] — 英文句子列表
        show_progress : bool — 是否显示进度

        Returns
        -------
        list[str] — 德文翻译列表
        """
        results = []
        n = len(sentences)

        for i, sent in enumerate(sentences):
            translation = self.translate(sent)
            results.append(translation)
            if show_progress and (i + 1) % 50 == 0:
                print(f'  [{i+1}/{n}]')

        return results


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def find_latest_experiment(phase=2):
    """自动查找最新的实验目录。"""
    exp_root = os.path.join(os.path.dirname(__file__), 'experiments')
    if not os.path.isdir(exp_root):
        return None

    dirs = [d for d in os.listdir(exp_root)
            if d.startswith(f'EXP-P{phase}-') and
            os.path.isdir(os.path.join(exp_root, d))]
    if not dirs:
        return None

    dirs.sort(reverse=True)  # 最新的在前
    return os.path.join(exp_root, dirs[0])


def interactive_mode(translator):
    """交互式翻译模式。"""
    print('=' * 60)
    print('交互式翻译模式 (beam_size={})'.format(translator.beam_size))
    print('输入英文句子，实时输出德文翻译。')
    print('输入 :q 或 :quit 退出，输入 :beam N 切换 beam size')
    print('=' * 60)
    print()

    while True:
        try:
            text = input('EN > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        if text in (':q', ':quit', ':exit'):
            break

        if text.startswith(':beam '):
            try:
                bs = int(text.split()[1])
                translator.beam_size = max(1, min(bs, 20))
                print(f'  beam_size → {translator.beam_size}')
            except (ValueError, IndexError):
                print('  用法: :beam <整数> (1-20)')
            continue

        translation = translator.translate(text)
        print(f'DE > {translation}')
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Transformer EN→DE 翻译推理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python predict.py --text "A man is running."
    python predict.py -i sentences.txt -o translations.txt
    python predict.py --interactive
    python predict.py --exp_dir experiments/EXP-P2-xxx --beam_size 5 -t "Hello."
        """,
    )

    # 模型
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='实验目录（含 config.json + checkpoint）')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt',
                        help='checkpoint 文件名 (默认 checkpoint_best.pt)')
    parser.add_argument('--phase', type=int, default=2,
                        help='自动查找最新 Phase N 实验 (默认 2)')

    # 解码
    parser.add_argument('--beam_size', type=int, default=1,
                        help='Beam Search 束宽 (1 = 贪心解码, 默认 1)')
    parser.add_argument('--max_len', type=int, default=100,
                        help='最大解码长度 (默认 100)')

    # 输入
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--text', '-t', type=str, default=None,
                             help='单句翻译')
    input_group.add_argument('--input', '-i', type=str, default=None,
                             help='输入文件路径（每行一句英文）')
    input_group.add_argument('--interactive', '-I', action='store_true',
                             help='交互模式')

    # 输出
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件路径（批量模式）')

    # 其他
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='强制指定设备')

    args = parser.parse_args()

    # ── 设备 ────────────────────────────────────────
    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else
                           'cpu'))

    # ── 查找实验目录 ────────────────────────────────
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = find_latest_experiment(args.phase)
        if exp_dir is None:
            print(f'未找到 Phase {args.phase} 的实验目录。'
                  f'请用 --exp_dir 手动指定。')
            sys.exit(1)
        print(f'自动选择: {exp_dir}\n')

    # ── 加载模型 ────────────────────────────────────
    model, src_vocab, tgt_vocab, config = load_model(
        exp_dir, args.checkpoint, device
    )

    translator = Translator(
        model, src_vocab, tgt_vocab,
        device=device,
        beam_size=args.beam_size,
        max_len=args.max_len,
    )

    # ── 执行翻译 ────────────────────────────────────
    if args.interactive:
        interactive_mode(translator)

    elif args.text:
        print(f'EN: {args.text}')
        translation = translator.translate(args.text)
        print(f'DE: {translation}')

    elif args.input:
        # 批量翻译
        input_path = args.input
        if not os.path.exists(input_path):
            print(f'输入文件不存在: {input_path}')
            sys.exit(1)

        with open(input_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        print(f'翻译 {len(sentences)} 句...')
        print(f'解码策略: {"Beam Search (k=" + str(args.beam_size) + ")" if args.beam_size > 1 else "Greedy"}')
        print()

        translations = translator.translate_batch(sentences)

        # 输出
        output_path = args.output
        if output_path is None:
            output_path = input_path.replace('.en.', '.de.').replace('.txt', '.de.txt')
            if output_path == input_path:
                output_path = input_path + '.de'

        with open(output_path, 'w', encoding='utf-8') as f:
            for t in translations:
                f.write(t + '\n')

        print(f'完成。输出: {output_path}')

        # 打印前几个样例
        print()
        print('─' * 60)
        print('样例 (前 5 条):')
        print('─' * 60)
        for i, (en, de) in enumerate(zip(sentences[:5], translations[:5])):
            print(f'[{i+1}] EN: {en}')
            print(f'    DE: {de}')
            print()

    else:
        # 无输入 → 交互模式
        interactive_mode(translator)


if __name__ == '__main__':
    main()
