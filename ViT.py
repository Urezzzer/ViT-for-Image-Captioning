import math

import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, dim, mlp_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        attn, attn_weights = self.attn(x2,x2,x2, need_weights=True)
        x = x + attn
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, dim: int, max_len: int, dropout: float = 0.1, mode='sincos'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        assert mode in {'sincos', 'params'}, 'mode should be sincos or params.'

        if mode == 'sincos':
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
            pe = torch.zeros(1, max_len, dim)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = nn.Parameter(torch.randn(1, max_len, dim))

        self.pe = pe

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderLayer(dim, mlp_dim, heads, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, pool=None, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_encoder = PositionalEncoding(dim, num_patches + 1, emb_dropout, mode='params')
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.pool = pool

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)

        x = self.transformer(x)

        if self.pool == 'cls':
            x = x[:, 0] # CLS token

        return x
