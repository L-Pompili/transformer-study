import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.scale * self._norm(x)

def precompute_rope_freqs(head_dim, max_len):
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    B, T, n_head, head_dim = x.shape

    cos = cos[:T].view(1, T, 1, head_dim // 2)
    sin = sin[:T].view(1, T, 1, head_dim // 2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, dim, bias=False) # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Value
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Output

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        cos, sin = precompute_rope_freqs(self.head_dim, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_len, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class ModernTinyTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            Block(config.d_model, config.n_head, config.d_ff, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])
        
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss