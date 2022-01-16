import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import numpy as np


class Image2Seq(nn.Module):
    """

    """
    def __init__(
            self,
            feature_extractor,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            attention_dim: int,
            encoder_output_dim: int,
            pad_idx: int,
            num_heads: int = 1,
            dropout: float = 0.1,
            init_decoder_weights_xavier: bool = True,
            init_embeds: bool = False,
            vectors: np.ndarray = None
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            feature_extractor,
            encoder_output_dim
        )
        self.decoder = Decoder(
            vocab_size,
            embedding_dim,
            hidden_dim,
            attention_dim,
            pad_idx,
            num_heads,
            dropout,
            encoder_output_dim)

        self.vocab_size = vocab_size

        if init_decoder_weights_xavier:
            for p in self.decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            warnings.warn('If not init_decoder_weights_xavier '
                          'embeddings weights are initialized from Uniform Distribution')

        if init_embeds:
            self.decoder.init_embeddings(vectors, pad_idx)
        else:
            warnings.warn('Embeddings weights are initialized from Xavier Uniform Distribution')

    def forward(self, images: Tensor, captions: Tensor,
                teacher_forcing_rate: float = 1, need_attn_weights: bool = True):

        features = self.encoder(images)
        preds, attention_weights = self.decoder(features, captions, teacher_forcing_rate)

        if need_attn_weights:
            return preds, attention_weights
        else:
            return preds


class Encoder(nn.Module):
    """

    """
    def __init__(
            self,
            feature_extractor,
            encoder_output_dim: int
    ) -> None:
        super(Encoder, self).__init__()

        self.feature_extractor = feature_extractor
        self.encoder_output_dim = encoder_output_dim

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        warnings.warn('Feature extractor is not trainable. If you want finetune it you need to fix this code yourself.')

    def forward(self, x):
        x = self.feature_extractor(x).view(x.shape[0],
                                           self.encoder_output_dim, -1).permute(0, 2, 1)
        # shape after == (batch_size, channels_dim, encoder_output_dim)
        return x


class Decoder(nn.Module):
    """

    """
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            attention_dim: int,
            pad_idx: int,
            num_heads: int = 1,
            dropout: float = 0.1,
            encoder_dim: int = 2048):
        super().__init__()

        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, kdim=encoder_dim, vdim=encoder_dim,
                                               batch_first=True)

        self.h0 = nn.Linear(encoder_dim, hidden_dim)
        self.c0 = nn.Linear(encoder_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)

        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: Tensor, captions: Tensor, teacher_forcing_rate: int = 1):

        embeds = self.dropout(self.embedding(captions))

        h, c = self.init_hidden_state(features)  # (batch_size, hidden_dim)
        seq_length = len(captions[0])
        batch_size = captions.size(0)
        num_features = features.size(1)
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(features.device)
        attention_weights = torch.zeros(batch_size, seq_length, num_features).to(features.device)

        for i in range(0, seq_length):
            if i == 0:
                input = embeds[:, 0] # (batch_size, embedding_dim)
            else:
                input = embeds[:, i] if np.random.uniform() < teacher_forcing_rate \
                    else self.embedding(output.view(batch_size, -1).argmax(dim=1).unsqueeze(0)).squeeze(0)

            context, attention_weight = self.attention(h.unsqueeze(1), features, features)
            # context shape = (batch_size, hidden_dim)

            h = self.layer_norm(h + context.squeeze())

            h, c = self.lstm_cell(input, (h, c))

            output = self.linear(h) # (batch_size, vocab_size)

            preds[:, i] = output
            attention_weights[:, i] = attention_weight.squeeze()

        # preds shape = (batch_size, seq_length, vocab_size)
        # attention_weights = (batch_size, seq_length, num_features)
        return preds, attention_weights

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = self.pool(encoder_out.permute(0, 2, 1)).squeeze(-1)
        mean_encoder_out = F.gelu(mean_encoder_out)
        h = self.h0(mean_encoder_out)  # (batch_size, hidden_dim)
        c = self.c0(mean_encoder_out)  # (batch_size, hidden_dim)

        return h, c

    def init_embeddings(self, vectors, pad_idx: int, freeze: bool = False) -> None:
        weights = torch.tensor(vectors, requires_grad=True, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze, padding_idx=pad_idx)
