import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat


class MIM(nn.Module):
    def __init__(
            self,
            *,
            encoder,
            masking_ratio=0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_encoder.pe.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device=device)[:, None]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = self.encoder.pos_encoder(tokens)

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_patches)
        mask_tokens = self.encoder.pos_encoder(mask_tokens)

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device=device).topk(k=num_masked, dim=-1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device=device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        return pred_pixel_values, masked_patches
