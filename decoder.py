import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

class CustomDecoderBlock(nn.Module):
    def __init__(self, d_model, heads, ff_dim):
        super().__init__()
        self.heads = heads
        self.d_model = d_model

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.cross_q_proj = nn.Linear(d_model, d_model)
        self.cross_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def split_heads(self, x):
        B, T, D = x.size()
        return x.view(B, T, self.heads, D // self.heads).transpose(1, 2)

    def combine_heads(self, x):
        B, H, T, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self, x, context, tgt_mask=None):
        # Self-attention
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        attn = scaled_dot_product_attention(q, k, v, tgt_mask)
        attn = self.combine_heads(attn)
        x = self.norm1(x + attn)

        # Cross-attention (image features)
        cq = self.cross_q_proj(x)
        ck, cv = self.cross_kv_proj(context).chunk(2, dim=-1)
        cq, ck, cv = self.split_heads(cq), self.split_heads(ck), self.split_heads(cv)
        cross_attn = scaled_dot_product_attention(cq, ck, cv)
        cross_attn = self.combine_heads(cross_attn)
        x = self.norm2(x + cross_attn)

        # Feedforward
        x = self.norm3(x + self.ff(x))
        return x

class FullCaptionDecoder(nn.Module):
    def __init__(self, clip_text_embedding, vocab_size, d_model=512, num_layers=2, heads=8, ff_dim=2048, max_len=40):
        super().__init__()
        self.token_embed = clip_text_embedding  # ← reusing CLIP's word embedding
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            CustomDecoderBlock(d_model, heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, memory, mask=None):
        x = self.token_embed(input_ids)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, context=memory, tgt_mask=mask)
        return self.output_layer(x)
