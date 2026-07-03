"""
Transformer 序列到序列模型。

组装顺序：
    token → Embedding → PositionalEncoding → Encoder ─→ memory
    token → Embedding → PositionalEncoding → Decoder ─→ hidden
                                                           │
                                        Generator ←────────┘
                                           │
                                      log_probs
"""
import copy
import torch
import torch.nn as nn

from common import (
    Decoder,
    DecoderLayer,
    Embedding,
    Encoder,
    EncoderLayer,
    FNN,
    Generator,
    MultiHeadAttention,
    PositionalEncoding,
)


class EncoderDecoder(nn.Module):
    """纯 Encoder-Decoder 结构。

    接收已嵌入的张量，返回解码器隐藏状态。
    不含 Embedding、PositionalEncoding、Generator。
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_embedded, tgt_embedded, src_mask, tgt_mask):
        """前向传播：编码 → 解码 → 隐藏状态"""
        memory = self.encode(src_embedded, src_mask)
        return self.decode(tgt_embedded, memory, src_mask, tgt_mask)

    def encode(self, src_embedded, src_mask):
        """仅编码：已嵌入源序列 → 编码器记忆"""
        return self.encoder(src_embedded, src_mask)

    def decode(self, tgt_embedded, memory, src_mask, tgt_mask):
        """仅解码：已嵌入目标序列 + 编码器记忆 → 隐藏状态"""
        return self.decoder(tgt_embedded, memory, src_mask, tgt_mask)


class Transformer(nn.Module):
    """完整的 Transformer 序列到序列模型。

    Parameters
    ----------
    src_vocab_size : int
        源词汇表大小。
    tgt_vocab_size : int
        目标词汇表大小。
    num_layers : int, default 6
        编码器和解码器的层数。
    d_model : int, default 512
        模型维度（嵌入维度）。
    num_heads : int, default 8
        多头注意力头数。
    d_ff : int, default 2048
        前馈网络隐藏层维度（标准为 4 × d_model）。
    dropout : float, default 0.1
        Dropout 概率。
    max_len : int, default 5000
        位置编码最大序列长度。
    padding_idx : int, default 0
        嵌入层填充索引。
    share_embed : bool, default False
        是否共享源和目标嵌入层（要求 src_vocab_size == tgt_vocab_size）。
    tie_embed_weights : bool, default False
        是否将目标嵌入层权重绑定到 Generator 输出投影层。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
        share_embed: bool = False,
        tie_embed_weights: bool = False,
    ):
        super().__init__()

        # ── 嵌入层 ──────────────────────────────────────────
        src_emb = Embedding(src_vocab_size, d_model, padding_idx)
        if share_embed and src_vocab_size == tgt_vocab_size:
            tgt_emb = src_emb  # 共享嵌入
        else:
            tgt_emb = Embedding(tgt_vocab_size, d_model, padding_idx)

        # PositionalEncoding 零参数模块，source 和 target 共享同一实例
        pos_enc = PositionalEncoding(d_model, dropout, max_len)
        self.source_embed = nn.Sequential(src_emb, pos_enc)
        self.target_embed = nn.Sequential(tgt_emb, pos_enc)

        # ── 编码器 / 解码器 ─────────────────────────────────
        attn = MultiHeadAttention(num_heads, d_model, dropout)
        ffn = FNN(d_model, d_ff, d_model, dropout)

        encoder_layer = EncoderLayer(d_model, attn, ffn, dropout)
        decoder_layer = DecoderLayer(
            d_model,
            copy.deepcopy(attn),   # decoder 自注意力
            copy.deepcopy(attn),   # 交叉注意力
            copy.deepcopy(ffn),
            dropout,
        )

        self.encoder_decoder = EncoderDecoder(
            Encoder(encoder_layer, num_layers),
            Decoder(decoder_layer, num_layers),
        )

        # ── 输出投影层（Generator 外置）─────────────────────
        self.generator = Generator(d_model, tgt_vocab_size)

        # ── 可选：权重绑定 ──────────────────────────────────
        if tie_embed_weights:
            self.generator.project.weight = self.target_embed[0].emb.weight

        # ── 参数初始化 ──────────────────────────────────────
        self._init_parameters()

    def _init_parameters(self):
        """逐模块参数初始化（预留扩展点）。"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """完整前向传播：嵌入 → 编码 → 解码 → 生成 → log_probs。

        Parameters
        ----------
        src : Tensor, shape (batch, src_seq)
            源序列 token ID。
        tgt : Tensor, shape (batch, tgt_seq)
            目标序列 token ID。
        src_mask : Tensor, optional
            源序列注意力掩码。
        tgt_mask : Tensor, optional
            目标序列注意力掩码。

        Returns
        -------
        Tensor, shape (batch, tgt_seq, tgt_vocab)
            log-概率分布。
        """
        src_emb = self.source_embed(src)
        tgt_emb = self.target_embed(tgt)
        hidden = self.encoder_decoder(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.generator(hidden)

    def encode(self, src, src_mask=None):
        """仅编码：源 token → memory（用于推理缓存）。"""
        return self.encoder_decoder.encode(self.source_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        """仅解码：目标 token + memory → 隐藏状态。"""
        return self.encoder_decoder.decode(
            self.target_embed(tgt), memory, src_mask, tgt_mask
        )


# ═══════════════════════════════════════════════════════════════════
# 冒烟测试
# ═══════════════════════════════════════════════════════════════════

def test_transformer():
    """轻量级冒烟测试：验证前向传播形状、log_softmax、权重绑定、组件共享。"""
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        num_layers=2,
        d_model=256,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        tie_embed_weights=True,
    )

    src = torch.randint(0, 1000, (2, 10))  # (batch=2, seq=10)
    tgt = torch.randint(0, 1000, (2, 8))   # (batch=2, seq=8)

    output = model(src, tgt)

    # ① 输出形状验证
    assert output.shape == (2, 8, 1000), (
        f"期望 (2,8,1000)，实际 {output.shape}"
    )

    # ② log_softmax 输出 ≤ 0
    assert (output <= 0).all(), "log_softmax 输出应 ≤ 0"

    # ③ 概率和为 1
    probs = torch.exp(output)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 8), atol=1e-5), (
        "概率和应接近 1"
    )

    # ④ 验证权重绑定
    assert model.generator.project.weight is model.target_embed[0].emb.weight, (
        "权重绑定失败：generator 和 target_embed 应共享同一权重矩阵"
    )

    # ⑤ 验证 PositionalEncoding 共享
    assert model.source_embed[1] is model.target_embed[1], (
        "PositionalEncoding 未共享：source_embed[1] 和 target_embed[1] 应为同一实例"
    )

    print(f"✓ 冒烟测试全部通过")
    print(f"  输出形状:  {output.shape}")
    print(f"  参数量:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"  log_softmax ≤ 0:  通过")
    print(f"  概率和 = 1:       通过")
    print(f"  权重绑定:         通过")
    print(f"  PositionalEncoding 共享: 通过")


if __name__ == "__main__":
    test_transformer()
