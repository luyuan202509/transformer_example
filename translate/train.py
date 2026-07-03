"""
Transformer 翻译模型训练脚本。

用法:
    python train.py                     # 默认 Phase 1 冒烟测试
    python train.py --phase 2           # Phase 2 基线训练
    python train.py --phase 3           # Phase 3 优化提升
    python train.py --phase 1 --epochs 10   # 自定义 epoch 数

实验阶段:
    Phase 1 — 冒烟测试: d_model=128, 2层, 500 steps, 验证正确性
    Phase 2 — 基线训练: d_model=256, 3层, 20 epochs, BLEU ≥ 15
    Phase 3 — 优化提升: d_model=512, 6层, 30 epochs, BLEU ≥ 25
    Phase 4 — 精调:    基于 Phase 3 checkpoint 继续调优
"""
import os
import sys
import json
import math
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加 model 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from transformer import Transformer

# 添加 dataset 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset'))
from load_data import load_multi30k, PAD_IDX, SOS_IDX, EOS_IDX


# ═══════════════════════════════════════════════════════════════════
# 学习率调度器
# ═══════════════════════════════════════════════════════════════════

class TransformerLR(torch.optim.lr_scheduler._LRScheduler):
    """Transformer Warmup + Inverse Sqrt Decay.

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.d_model ** (-0.5)
        arg = min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [scale * arg for _ in self.optimizer.param_groups]


# ═══════════════════════════════════════════════════════════════════
# 阶段配置
# ═══════════════════════════════════════════════════════════════════

PHASE_CONFIGS = {
    1: {  # 冒烟测试 — 验证正确性
        'num_layers': 2,
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 256,
        'dropout': 0.1,
        'batch_size': 32,
        'warmup_steps': 200,
        'max_epochs': 5,
        'label_smoothing': 0.0,       # Phase 1 不使用，先验证基础训练
        'tie_embed_weights': True,
        'grad_clip': 1.0,
        'data_limit': 5000,            # 只用前 5000 句快速迭代
        'target_bleu': None,           # Phase 1 不要求 BLEU
        'description': '冒烟测试: 验证训练流程正确性，Loss 持续下降即可',
    },
    2: {  # 基线训练
        'num_layers': 3,
        'd_model': 256,
        'num_heads': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'batch_size': 64,
        'warmup_steps': 2000,
        'max_epochs': 20,
        'label_smoothing': 0.1,
        'tie_embed_weights': True,
        'grad_clip': 1.0,
        'data_limit': None,            # 全量数据
        'target_bleu': 15,
        'description': '基线训练: 3层 256维，目标 BLEU ≥ 15',
    },
    3: {  # 优化提升
        'num_layers': 4,
        'd_model': 256,
        'num_heads': 8,
        'd_ff': 1024,
        'dropout': 0.15,               # 适度正则化
        'batch_size': 64,
        'warmup_steps': 3000,
        'max_epochs': 25,
        'label_smoothing': 0.1,
        'tie_embed_weights': True,
        'grad_clip': 1.0,
        'data_limit': None,
        'target_bleu': 20,
        'early_stopping_patience': 7,
        'description': 'Phase 3: 4层 256维 d_ff=1024 dropout=0.15 早停',
    },
    4: {  # 精调
        'num_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'dropout': 0.2,                # 增大 dropout 防过拟合
        'batch_size': 128,
        'warmup_steps': 4000,
        'max_epochs': 10,              # 在 Phase 3 基础上继续
        'label_smoothing': 0.15,
        'tie_embed_weights': True,
        'grad_clip': 1.0,
        'data_limit': None,
        'target_bleu': 30,
        'description': '精调: 正则化 + 推理优化，目标 BLEU ≥ 30',
    },
}


# ═══════════════════════════════════════════════════════════════════
# 掩码构造
# ═══════════════════════════════════════════════════════════════════

def create_masks(src, tgt, pad_idx=PAD_IDX):
    """构造 Encoder 和 Decoder 所需的注意力掩码。

    Returns
    -------
    src_mask : Tensor, shape (batch, 1, 1, src_len)
        源序列 Padding Mask — Encoder 自注意力用。
    tgt_mask : Tensor, shape (batch, 1, tgt_len, tgt_len)
        目标序列 Padding + Causal Mask — Decoder 自注意力用。
    """
    device = src.device

    # Encoder: 仅 Padding Mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Decoder: Padding Mask + Causal Mask
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=device), diagonal=1
    ) == 0
    tgt_mask = tgt_pad_mask & causal_mask

    return src_mask, tgt_mask


# ═══════════════════════════════════════════════════════════════════
# 单个训练 Step
# ═══════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失 — 与 log_softmax 输出配合使用。

    无平滑 (smoothing=0): 等价于 NLLLoss(ignore_index=pad_idx)
    有平滑 (smoothing=0.1): 真实 token 概率 = 1-ε，剩余 ε 平分给其他 token
    """

    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, log_probs, labels):
        """
        log_probs: (N, vocab_size) — log_softmax 输出
        labels:    (N,)          — 真实 token ID
        """
        if self.smoothing <= 0:
            return F.nll_loss(log_probs, labels, ignore_index=self.pad_idx)

        n_class = self.vocab_size - 1  # exclude padding
        smooth = self.smoothing / n_class

        true_dist = torch.full_like(log_probs, smooth)
        true_dist.scatter_(1, labels.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0

        # mask padding positions
        mask = (labels != self.pad_idx).float().unsqueeze(1)
        loss = -(true_dist * log_probs).sum(dim=-1) * mask.squeeze(1)
        return loss.sum() / mask.sum()


def train_step(model, src, tgt, criterion, optimizer, scheduler, grad_clip):
    """执行一个训练 step：Teacher Forcing + 前向 + 反向 + 更新。

    Parameters
    ----------
    src : (batch, src_seq) — 源序列（含 <s> 和 </s>）
    tgt : (batch, tgt_seq) — 目标序列（含 <s> 和 </s>）

    Returns
    -------
    loss.item() : float
    """
    model.train()

    # ── 1. Teacher Forcing ──────────────────────────
    tgt_input  = tgt[:, :-1]   # decoder 输入: 去掉 </s>
    tgt_output = tgt[:, 1:]    # 预测目标:    去掉 <s>

    # ── 2. 掩码 ─────────────────────────────────────
    src_mask, tgt_mask = create_masks(src, tgt_input)

    # ── 3. 前向传播 ─────────────────────────────────
    log_probs = model(src, tgt_input, src_mask, tgt_mask)
    # log_probs: (batch, tgt_seq-1, vocab_size)

    # ── 4. 损失 ─────────────────────────────────────
    loss = criterion(
        log_probs.reshape(-1, log_probs.size(-1)),   # (batch*seq, vocab)
        tgt_output.reshape(-1)                        # (batch*seq,)
    )

    # ── 5. 反向传播 ─────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip
    )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item(), grad_norm.item()


# ═══════════════════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, data_loader, criterion, max_batches=None):
    """在验证/测试集上计算损失和 token 准确率。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    n_batches = 0

    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask = create_masks(src, tgt_input)

        log_probs = model(src, tgt_input, src_mask, tgt_mask)

        loss = criterion(
            log_probs.reshape(-1, log_probs.size(-1)),
            tgt_output.reshape(-1)
        )

        # Token 准确率
        pred = log_probs.argmax(dim=-1)             # (batch, seq)
        non_pad = tgt_output != PAD_IDX
        correct_tokens += (pred == tgt_output)[non_pad].sum().item()
        total_tokens += non_pad.sum().item()

        total_loss += loss.item()
        n_batches += 1

        if max_batches and n_batches >= max_batches:
            break

    avg_loss = total_loss / n_batches
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss)

    return avg_loss, accuracy, perplexity


# ═══════════════════════════════════════════════════════════════════
# 推理解码
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    """自回归贪心解码：逐 token 生成翻译，不依赖 Teacher Forcing。

    Parameters
    ----------
    model : Transformer
    src : Tensor, shape (batch, src_seq) — 源序列 token ID
    src_mask : Tensor, shape (batch, 1, 1, src_seq)
    max_len : int — 最大生成长度
    start_symbol : int — <s> token ID
    end_symbol : int — </s> token ID

    Returns
    -------
    Tensor, shape (batch, generated_seq) — 生成的 token ID 序列
    """
    model.eval()
    memory = model.encode(src, src_mask)
    batch_size = src.size(0)
    device_local = src.device

    # 初始输入：只有 <s>
    ys = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device_local)

    # 记录哪些序列已经生成 </s>
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device_local)

    for _ in range(max_len - 1):
        # 为当前已生成序列构造因果掩码
        _, tgt_mask = create_masks(src, ys)

        # 解码整条序列，只取最后一个位置的输出
        decoded = model.decode(ys, memory, src_mask, tgt_mask)
        log_probs = model.generator(decoded[:, -1:, :])
        next_token = log_probs.argmax(dim=-1)  # (batch, 1)

        ys = torch.cat([ys, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == end_symbol)

        if finished.all():
            break

    return ys


# ═══════════════════════════════════════════════════════════════════
# BLEU 计算
# ═══════════════════════════════════════════════════════════════════

def compute_corpus_bleu(hypotheses, references):
    """计算语料库级别 BLEU-4 分数。

    优先使用 sacrebleu（标准实现），不可用时回退到简单实现。

    Parameters
    ----------
    hypotheses : list[str] — 模型生成的翻译
    references : list[str] — 参考翻译

    Returns
    -------
    float — BLEU-4 分数 (0-100)
    """
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score
    except ImportError:
        pass

    # ── 简单回退实现 ──────────────────────────────────
    from collections import Counter as _Counter

    def _get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    max_order = 4
    weights = [1.0 / max_order] * max_order

    # 简短惩罚
    hyp_len = sum(len(h.split()) for h in hypotheses)
    ref_len = sum(len(r.split()) for r in references)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)

    log_bleu = 0.0
    for n in range(1, max_order + 1):
        ref_counts = _Counter()
        for ref in references:
            ref_counts.update(_get_ngrams(ref.split(), n))

        clipped_count = 0
        total_count = 0
        for hyp in hypotheses:
            hyp_ngrams = _get_ngrams(hyp.split(), n)
            total_count += len(hyp_ngrams)
            for ng, count in _Counter(hyp_ngrams).items():
                clipped_count += min(count, ref_counts.get(ng, 0))

        if total_count == 0:
            return 0.0
        pn = clipped_count / total_count
        if pn == 0:
            return 0.0
        log_bleu += weights[n - 1] * math.log(pn)

    return bp * math.exp(log_bleu) * 100


@torch.no_grad()
def evaluate_bleu(model, data_loader, tgt_vocab, max_samples=200, max_len=100):
    """在数据集上运行贪心解码并计算 BLEU。

    每次只评测 max_samples 条样本，控制推理耗时。

    Parameters
    ----------
    model : Transformer
    data_loader : DataLoader
    tgt_vocab : Vocabulary — 将 token ID 解码为文本
    max_samples : int — 最多评估的样本数
    max_len : int — 最大解码长度（目标语言序列长度上限）

    Returns
    -------
    bleu_score : float
    """
    model.eval()
    hypotheses = []
    references = []
    sample_count = 0

    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)

        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

        generated = greedy_decode(model, src, src_mask, max_len, SOS_IDX, EOS_IDX)

        for i in range(src.size(0)):
            hyp = tgt_vocab.decode(generated[i].tolist())
            ref = tgt_vocab.decode(tgt[i].tolist())
            if hyp.strip():
                hypotheses.append(hyp)
                references.append(ref)

            sample_count += 1
            if sample_count >= max_samples:
                break

        if sample_count >= max_samples:
            break

    if len(hypotheses) == 0:
        return 0.0

    return compute_corpus_bleu(hypotheses, references)


# ═══════════════════════════════════════════════════════════════════
# 主训练循环
# ═══════════════════════════════════════════════════════════════════

def train(config, phase, exp_dir, resume_from=None):
    """执行完整训练流程。

    Parameters
    ----------
    config : dict — 超参数配置
    phase : int — 实验阶段 (1-4)
    exp_dir : str — 实验输出目录
    resume_from : str or None — 恢复训练的 checkpoint 路径
    """
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')

    print('=' * 60)
    print(f'Phase {phase}: {config["description"]}')
    print(f'设备: {device}')
    print(f'实验目录: {exp_dir}')
    print('=' * 60)

    # ── 加载数据 ────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'multi30k')

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        load_multi30k(data_dir, batch_size=config['batch_size'])

    # Phase 1: 子集数据快速迭代
    if config.get('data_limit'):
        train_loader.dataset.src_sentences = \
            train_loader.dataset.src_sentences[:config['data_limit']]
        train_loader.dataset.tgt_sentences = \
            train_loader.dataset.tgt_sentences[:config['data_limit']]
        print(f'数据子集: {config["data_limit"]} 句对')

    print(f'源词表: {len(src_vocab):,}  |  目标词表: {len(tgt_vocab):,}')
    print()

    # ── 创建模型 ────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_len=5000,
        padding_idx=PAD_IDX,
        tie_embed_weights=config['tie_embed_weights'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数: {n_params:,}  (可训练: {n_trainable:,})')
    print(f'd_model={config["d_model"]}, layers={config["num_layers"]}, '
          f'heads={config["num_heads"]}, d_ff={config["d_ff"]}')
    print()

    # ── 恢复 checkpoint ────────────────────────────
    start_epoch = 1
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'从 {resume_from} 恢复，从 epoch {start_epoch} 继续')
        print()

    # ── 优化器 ──────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,                          # 实际 lr 由 scheduler 控制
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = TransformerLR(
        optimizer,
        d_model=config['d_model'],
        warmup_steps=config['warmup_steps'],
    )

    # ── 损失函数 ────────────────────────────────────
    smoothing = config.get('label_smoothing', 0.0)
    if smoothing > 0:
        criterion = LabelSmoothingLoss(
            len(tgt_vocab), PAD_IDX, smoothing=smoothing
        )
        print(f'标签平滑: ε={smoothing}')
    else:
        criterion = nn.NLLLoss(ignore_index=PAD_IDX)
    print()

    # ── 实验记录 ────────────────────────────────────
    os.makedirs(exp_dir, exist_ok=True)
    metrics_log = []
    best_val_loss = float('inf')
    best_val_bleu = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', None)
    total_steps = 0

    # 保存配置
    config_to_save = {k: str(v) if isinstance(v, torch.device) else v
                      for k, v in config.items()}
    config_to_save['src_vocab_size'] = len(src_vocab)
    config_to_save['tgt_vocab_size'] = len(tgt_vocab)
    config_to_save['n_params'] = n_params
    config_to_save['device'] = str(device)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    # ═════════════════════════════════════════════════
    # 训练循环
    # ═════════════════════════════════════════════════
    print('开始训练...')
    print(f'{"Epoch":>5} {"Step":>6} {"Train Loss":>10} '
          f'{"Val Loss":>10} {"PPL":>8} {"Acc":>7} {"BLEU":>7} {"LR":>10} {"Time":>8}')
    print('-' * 84)

    t_start = time.time()

    for epoch in range(start_epoch, config['max_epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            loss, grad_norm = train_step(
                model, src, tgt, criterion, optimizer,
                scheduler, config['grad_clip']
            )

            epoch_loss += loss
            n_batches += 1
            total_steps += 1

            # 每 100 step 打印
            if total_steps % 100 == 0:
                elapsed = time.time() - t_start
                current_lr = scheduler.get_last_lr()[0]
                print(f'{epoch:5d} {total_steps:6d} {loss:10.4f} '
                      f'{"-":>10} {"-":>8} {"-":>7} {"-":>7} '
                      f'{current_lr:10.6f} {elapsed:7.0f}s')

            # Phase 1 提前停止条件：steps 够了
            if phase == 1 and total_steps >= 500:
                break

        # ── 每个 epoch 结束：验证 ──────────────────
        val_loss, val_acc, val_ppl = validate(model, val_loader, criterion)

        # BLEU 评测（推理阶段，不依赖 Teacher Forcing）
        bleu_samples = 200
        val_bleu = evaluate_bleu(
            model, val_loader, tgt_vocab, max_samples=bleu_samples, max_len=100
        )

        avg_train_loss = epoch_loss / n_batches
        elapsed = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]

        print(f'{epoch:5d} {total_steps:6d} {avg_train_loss:10.4f} '
              f'{val_loss:10.4f} {val_ppl:8.2f} {val_acc:7.2%} '
              f'{val_bleu:6.1f} '
              f'{current_lr:10.6f} {elapsed:7.0f}s')

        # ── 记录指标 ──────────────────────────────
        metrics = {
            'epoch': epoch,
            'total_steps': total_steps,
            'train_loss': round(avg_train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_ppl': round(val_ppl, 2),
            'val_accuracy': round(val_acc, 4),
            'val_bleu': round(val_bleu, 1),
            'lr': current_lr,
            'elapsed_seconds': round(elapsed, 1),
        }
        metrics_log.append(metrics)

        # ── 保存最佳模型（以 BLEU 为主要指标）───
        is_best = val_bleu > best_val_bleu if best_val_bleu > 0 else val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_bleu = val_bleu
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'total_steps': total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'val_bleu': val_bleu,
                'config': config,
            }, os.path.join(exp_dir, 'checkpoint_best.pt'))
            print(f'  → 保存最佳模型 (val_loss={val_loss:.4f}, BLEU={val_bleu:.1f})')
        else:
            patience_counter += 1

        # ── Phase 1 早停 ───────────────────────────
        if phase == 1 and total_steps >= 500:
            print()
            print(f'Phase 1 完成: {total_steps} steps, '
                  f'初始 loss 约 7-8, 最终 val_loss={val_loss:.4f}, BLEU={val_bleu:.1f}')
            if val_loss < 5.0:
                print('✓ 通过！Loss 已降到 5 以下，训练流程正常。')
                print('  可以进入 Phase 2: python train.py --phase 2')
            else:
                print('⚠ val_loss 仍然较高，建议增加 steps 或检查数据')
            break

        # ── 早停检查 ───────────────────────────────
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print()
            print(f'早停触发: val_loss 连续 {early_stopping_patience} 个 epoch 未改善')
            print(f'最佳 epoch: {best_epoch}, 最佳 val_loss: {best_val_loss:.4f}, '
                  f'最佳 BLEU: {best_val_bleu:.1f}')
            break

    # ── 保存最终模型和指标 ────────────────────────
    torch.save({
        'epoch': epoch,
        'total_steps': total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(exp_dir, 'checkpoint_last.pt'))

    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=2, ensure_ascii=False)

    # ── 最终测试（Phase 2+） ──────────────────────
    if phase >= 2:
        print()
        print('在测试集上评估...')
        test_loss, test_acc, test_ppl = validate(
            model, test_loader, criterion
        )
        print(f'测试集 (Teacher Forcing): Loss={test_loss:.4f}  '
              f'PPL={test_ppl:.2f}  Acc={test_acc:.2%}')

        # 加载最佳 checkpoint 再做推理评估
        best_ckpt_path = os.path.join(exp_dir, 'checkpoint_best.pt')
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print('  加载最佳模型进行 BLEU 评测...')

        test_bleu = evaluate_bleu(
            model, test_loader, tgt_vocab, max_samples=500, max_len=100
        )
        print(f'测试集 (推理): BLEU={test_bleu:.1f}')

        # 保存测试结果
        test_results = {
            'test_loss': round(test_loss, 4),
            'test_ppl': round(test_ppl, 2),
            'test_accuracy': round(test_acc, 4),
            'test_bleu': round(test_bleu, 1),
        }
        with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)

    print()
    total_time = time.time() - t_start
    print(f'训练完成。总耗时: {total_time/60:.1f} 分钟')
    print(f'最佳 val_loss: {best_val_loss:.4f}  最佳 BLEU: {best_val_bleu:.1f}')
    print(f'实验目录: {exp_dir}')


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transformer 翻译模型训练'
    )
    parser.add_argument('--phase', type=int, default=1,
                        help='实验阶段 (1-4), 默认 1')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖默认 epoch 数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖默认 batch_size')
    parser.add_argument('--d_model', type=int, default=None,
                        help='覆盖默认 d_model')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='覆盖默认 num_layers')
    parser.add_argument('--dropout', type=float, default=None,
                        help='覆盖默认 dropout')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的 checkpoint 路径')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='自定义实验名称')
    args = parser.parse_args()

    # ── 加载配置 ────────────────────────────────────
    if args.phase not in PHASE_CONFIGS:
        print(f'无效阶段: {args.phase}，请选择 1-4')
        sys.exit(1)

    config = PHASE_CONFIGS[args.phase].copy()

    # 命令行覆盖
    if args.epochs is not None:
        config['max_epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.d_model is not None:
        config['d_model'] = args.d_model
    if args.num_layers is not None:
        config['num_layers'] = args.num_layers
    if args.dropout is not None:
        config['dropout'] = args.dropout

    # ── 实验目录 ────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.exp_name:
        exp_dir = os.path.join(
            os.path.dirname(__file__), 'experiments', args.exp_name
        )
    else:
        exp_dir = os.path.join(
            os.path.dirname(__file__), 'experiments',
            f'EXP-P{args.phase}-{timestamp}'
        )

    # ── 开始训练 ────────────────────────────────────
    train(config, args.phase, exp_dir, resume_from=args.resume)
